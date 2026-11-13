"""The discovery loop: seed -> citation graph + keyword search -> rank -> assess -> next round.

Each round expands from the papers the previous round accepted, so the search
follows what it found rather than its first keywords. The LLM only ever writes
queries and scores real records; it never names a paper.
"""

import hashlib
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field

from tb_search.assess import _reply_text, assess
from tb_search.prompts import QUERY_PROMPT
from tb_search.screening import _wait_out_rate_limit
from tb_search.sources import Paper
from tb_search.store import Store
from tb_search.verdicts import parse_string_list

TOKEN = re.compile(r"[a-z][a-z0-9-]{2,}")
STOPWORDS = frozenset({
    "the", "and", "for", "with", "from", "that", "this", "these", "those", "are", "was", "were",
    "been", "has", "have", "had", "not", "but", "can", "may", "will", "its", "their", "our",
    "into", "than", "then", "also", "using", "used", "use", "based", "via", "between", "among",
    "across", "within", "over", "under", "more", "most", "such", "both", "each", "other",
    "which", "while", "where", "when", "how", "what", "who", "paper", "papers", "find",
    "relevant", "like", "related", "study", "studies", "results", "show", "shows", "new",
    "novel", "approach", "analysis", "data", "http", "https", "www", "org", "com", "content",
    "full",
})

# Every candidate a round turns up gets its abstract fetched before ranking;
# past this many, only the best by title are worth the requests.
HYDRATE_CAP = 300


@dataclass
class DiscoveryConfig:
    rounds: int = 2
    # LLM assessments per round — the knob that sizes a run to a TPM budget.
    budget: int = 20
    max_papers: int = 10
    per_query: int = 25
    queries_per_round: int = 3
    # Accepted papers whose references/citations seed the next round.
    expand: int = 5
    # Stop when fewer than this share of a round's assessed papers are accepted.
    min_yield: float = 0.15
    accept: int = 2
    # Off for evaluation, where the seed's reference list is the held-out gold set.
    use_seed_references: bool = True


@dataclass
class Candidate:
    paper: Paper
    round: int
    routes: list[str] = field(default_factory=list)
    score: float = 0.0
    assessment: dict | None = None

    @property
    def relevance(self) -> int | None:
        return self.assessment["relevance"] if self.assessment else None


@dataclass
class Run:
    seed: Paper | None
    topic: str
    assessed: bool
    candidates: dict[str, Candidate] = field(default_factory=dict)
    queries: list[str] = field(default_factory=list)

    def ranked(self) -> list[Candidate]:
        """Every candidate, best first: accepted, then unjudged, then rejected."""

        def order(candidate: Candidate):
            relevance = candidate.relevance
            tier = 1.5 if relevance is None else float(relevance)
            return (tier, candidate.score)

        return sorted(self.candidates.values(), key=order, reverse=True)

    def papers(self, max_papers: int, accept: int) -> list[Candidate]:
        ranked = self.ranked()
        if self.assessed:
            ranked = [c for c in ranked if c.relevance is not None and c.relevance >= accept]
        # A preprint and its published version are two records with one title.
        unique: dict[str, Candidate] = {}
        for candidate in ranked:
            unique.setdefault(" ".join(tokens(candidate.paper.title)), candidate)
        return list(unique.values())[:max_papers]


def tokens(text: str) -> list[str]:
    """Content words in order, deduplicated."""
    seen: dict[str, None] = {}
    for token in TOKEN.findall(text.lower()):
        if token not in STOPWORDS:
            seen.setdefault(token)
    return list(seen)


def keyword_query(text: str, limit: int = 12) -> str:
    """An OR query over the text's content words; Europe PMC ranks best matches first."""
    return " OR ".join(tokens(text)[:limit])


def lexical_scores(topic: str, papers: list[Paper]) -> dict[str, float]:
    """Share of the topic's idf-weighted vocabulary each paper covers, in [0, 1]."""
    wanted = set(tokens(topic))
    docs = {paper.key: set(tokens(f"{paper.title} {paper.abstract}")) for paper in papers}
    frequency = Counter(token for doc in docs.values() for token in doc)

    def idf(token: str) -> float:
        return math.log((len(docs) + 1) / (frequency[token] + 1)) + 1

    total = sum(idf(token) for token in wanted) or 1.0
    return {key: sum(idf(token) for token in doc & wanted) / total for key, doc in docs.items()}


def generate_queries(llm, topic: str, found: list[str], used: list[str], count: int) -> list[str]:
    """Ask the LLM for search queries from new angles; [] when it gives nothing usable."""
    prompt = (
        QUERY_PROMPT.replace("{TOPIC}", topic)
        .replace("{FOUND}", "\n".join(f"- {title}" for title in found) or "(none yet)")
        .replace("{USED}", "\n".join(f"- {query}" for query in used) or "(none yet)")
        .replace("{N}", str(count))
    )
    try:
        reply = _wait_out_rate_limit(llm.invoke, prompt)
    except Exception as e:  # noqa: BLE001 - the graph routes still work without new queries
        print(f"  ! query generation failed: {e}", flush=True)
        return []
    taken = {query.lower() for query in used}
    queries = []
    for query in parse_string_list(_reply_text(reply)):
        if query.lower() not in taken:
            taken.add(query.lower())
            queries.append(query)
    return queries[:count]


def _topic_key(seed: Paper | None, query: str) -> str:
    return seed.key if seed else hashlib.sha1(query.encode()).hexdigest()[:16]


def discover(
    source,
    store: Store,
    query: str,
    llm=None,
    model: str = "",
    config: DiscoveryConfig | None = None,
    log=print,
) -> Run:
    """Run the loop for one request. Without an llm, ranking is lexical only."""
    config = config or DiscoveryConfig()
    seed = source.resolve(query)
    if seed:
        source.hydrate([seed])
        topic = f"{seed.title}. {seed.abstract}".strip()
        log(f"  seed: {seed.title} ({seed.year}) [{seed.key}]")
    else:
        topic = query
        log("  no seed paper in the query; treating it as a topic")

    run = Run(seed=seed, topic=topic, assessed=llm is not None)
    run_id = store.start_run(query, seed.key if seed else "", asdict(config))
    topic_key = _topic_key(seed, query)
    store.event(run_id, 0, "seed", {"key": seed.key if seed else None, "title": topic[:200]})

    frontier = [seed] if seed else []
    queries = [keyword_query(seed.title if seed else query)]
    if llm:
        queries += generate_queries(llm, topic[:1500], [], queries, config.queries_per_round)

    def lookup(action, argument, route: str, kind: str, fresh: dict, round_number: int) -> None:
        """One source call; its papers join this round's fresh candidates."""
        try:
            papers = action(argument)
        except Exception as e:  # noqa: BLE001 - one failed lookup must not end the run
            log(f"  [{round_number}] ! {route} failed: {e}")
            return
        store.event(run_id, round_number, kind, {"route": route, "found": len(papers)})
        log(f"  [{round_number}] {route} -> {len(papers)}")
        for paper in papers:
            if seed and paper.key == seed.key:
                continue
            # Reached again by another path: a signal, not a new paper.
            candidate = run.candidates.get(paper.key) or fresh.setdefault(
                paper.key, Candidate(paper, round_number)
            )
            if route not in candidate.routes:
                candidate.routes.append(route)

    def search(text: str) -> list[Paper]:
        return source.search(text, config.per_query)

    for round_number in range(1, config.rounds + 1):
        fresh: dict[str, Candidate] = {}

        for paper in frontier:
            if paper is not seed or config.use_seed_references:
                lookup(
                    source.references, paper, f"cited-by:{paper.key}", "expand",
                    fresh, round_number,
                )
            lookup(source.citations, paper, f"cites:{paper.key}", "expand", fresh, round_number)
        for text in queries:
            run.queries.append(text)
            lookup(search, text, f"search:{text}", "search", fresh, round_number)

        if not fresh:
            store.event(run_id, round_number, "stop", {"reason": "no new candidates"})
            log(f"  [{round_number}] no new candidates; stopping")
            break

        pool = list(fresh.values())
        if len(pool) > HYDRATE_CAP:
            by_title = lexical_scores(topic, [c.paper for c in pool])
            pool.sort(key=lambda c: by_title[c.paper.key], reverse=True)
        try:
            source.hydrate([c.paper for c in pool[:HYDRATE_CAP]])
        except Exception as e:  # noqa: BLE001 - rank on titles if abstracts can't be had
            log(f"  [{round_number}] ! abstract fetch failed: {e}")

        scores = lexical_scores(topic, [c.paper for c in pool])
        for candidate in pool:
            # Reached by several routes (co-citation, search + graph): boost.
            bonus = 1 + 0.25 * min(len(candidate.routes) - 1, 4)
            candidate.score = round(scores[candidate.paper.key] * bonus, 4)
        pool.sort(key=lambda c: c.score, reverse=True)
        run.candidates.update(fresh)

        shortlist = pool[: config.budget]
        if llm:
            for candidate in shortlist:
                paper = candidate.paper
                cached = store.cached_assessment(paper.key, topic_key, model)
                if cached is None:
                    text = f"{paper.title}. {paper.abstract}".strip()
                    try:
                        cached = assess(llm, paper.title, text, topic[:1500])
                    except Exception as e:  # noqa: BLE001 - leave this one unjudged
                        log(f"  [{round_number}] ! assessment failed: {e}")
                    if cached:
                        store.save_assessment(paper.key, topic_key, model, cached)
                candidate.assessment = cached
                mark = "?" if candidate.relevance is None else candidate.relevance
                log(f"  [{round_number}] [{mark}] {paper.title[:90]}")
            accepted = [
                c for c in shortlist if c.relevance is not None and c.relevance >= config.accept
            ]
        else:
            accepted = shortlist[: config.expand]

        for candidate in pool:
            store.save_paper(candidate.paper)
            store.save_candidate(
                run_id, candidate.paper.key, round_number, candidate.routes,
                candidate.score, candidate.relevance,
            )
        share = len(accepted) / len(shortlist)
        store.event(
            run_id, round_number, "round",
            {"candidates": len(pool), "judged": len(shortlist), "accepted": len(accepted)},
        )
        log(
            f"  [{round_number}] {len(pool)} new candidates, "
            f"{len(shortlist)} judged, {len(accepted)} accepted"
        )

        if llm and share < config.min_yield:
            store.event(run_id, round_number, "stop", {"reason": "yield below threshold"})
            log(f"  [{round_number}] yield {share:.0%} is under {config.min_yield:.0%}; stopping")
            break

        frontier = [c.paper for c in accepted[: config.expand]]
        queries = []
        if llm and round_number < config.rounds:
            queries = generate_queries(
                llm, topic[:1500], [c.paper.title for c in accepted[:8]],
                run.queries, config.queries_per_round,
            )

    return run
