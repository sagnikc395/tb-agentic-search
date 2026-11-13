"""The discovery loop: seed -> citation graph + keyword search -> assess -> next round.

Each round expands from the papers the previous round accepted, so the search
follows what it found rather than its first keywords. Ranking is structural —
a paper several routes reached is judged before one only a single search found —
and the LLM only writes queries and scores records; it never names a paper.
"""

import re
from dataclasses import dataclass, field

from tb_search.assess import _reply_text, assess
from tb_search.prompts import QUERY_PROMPT
from tb_search.screening import _wait_out_rate_limit
from tb_search.sources import Paper
from tb_search.verdicts import parse_string_list

TOKEN = re.compile(r"[a-z][a-z0-9-]{2,}")
STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "these",
        "those",
        "are",
        "was",
        "were",
        "been",
        "has",
        "have",
        "had",
        "not",
        "but",
        "can",
        "may",
        "will",
        "its",
        "their",
        "our",
        "into",
        "than",
        "then",
        "also",
        "using",
        "used",
        "use",
        "based",
        "via",
        "between",
        "among",
        "across",
        "within",
        "over",
        "under",
        "more",
        "most",
        "such",
        "both",
        "each",
        "other",
        "which",
        "while",
        "where",
        "when",
        "how",
        "what",
        "who",
        "paper",
        "papers",
        "find",
        "relevant",
        "like",
        "related",
        "study",
        "studies",
        "results",
        "show",
        "shows",
        "new",
        "novel",
        "approach",
        "analysis",
        "data",
        "http",
        "https",
        "www",
        "org",
        "com",
        "content",
        "full",
    }
)

# Every candidate a round turns up gets its abstract fetched before the LLM
# sees it; past this many, only the best-connected are worth the requests.
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
    assessment: dict | None = None

    @property
    def relevance(self) -> int | None:
        return self.assessment["relevance"] if self.assessment else None

    @property
    def support(self) -> tuple[int, int]:
        """Structural rank: paths that reached this paper, then its citation count."""
        return (len(self.routes), self.paper.cited_by)


@dataclass
class Run:
    seed: Paper | None
    topic: str
    candidates: dict[str, Candidate] = field(default_factory=dict)
    queries: list[str] = field(default_factory=list)

    def ranked(self) -> list[Candidate]:
        """Every candidate, best first: accepted, then unjudged, then rejected."""

        def order(candidate: Candidate):
            relevance = candidate.relevance
            tier = 1.5 if relevance is None else float(relevance)
            return (tier, *candidate.support)

        return sorted(self.candidates.values(), key=order, reverse=True)

    def papers(self, max_papers: int, accept: int) -> list[Candidate]:
        ranked = [
            candidate
            for candidate in self.ranked()
            if candidate.relevance is not None and candidate.relevance >= accept
        ]
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


def generate_queries(
    llm, topic: str, found: list[str], used: list[str], count: int
) -> list[str]:
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


def discover(
    source,
    query: str,
    llm,
    config: DiscoveryConfig | None = None,
    log=print,
) -> Run:
    """Run the loop for one request; the LLM judges every candidate it is shown."""
    config = config or DiscoveryConfig()
    seed = source.resolve(query)
    if seed:
        source.hydrate([seed])
        topic = f"{seed.title}. {seed.abstract}".strip()
        log(f"  seed: {seed.title} ({seed.year}) [{seed.key}]")
    else:
        topic = query
        log("  no seed paper in the query; treating it as a topic")

    run = Run(seed=seed, topic=topic)

    frontier = [seed] if seed else []
    queries = [keyword_query(seed.title if seed else query)]
    queries += generate_queries(llm, topic[:1500], [], queries, config.queries_per_round)

    def lookup(action, argument, route: str, fresh: dict, round_number: int) -> None:
        """One source call; its papers join this round's fresh candidates."""
        try:
            papers = action(argument)
        except Exception as e:  # noqa: BLE001 - one failed lookup must not end the run
            log(f"  [{round_number}] ! {route} failed: {e}")
            return
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
                    source.references,
                    paper,
                    f"cited-by:{paper.key}",
                    fresh,
                    round_number,
                )
            lookup(
                source.citations,
                paper,
                f"cites:{paper.key}",
                fresh,
                round_number,
            )
        for text in queries:
            run.queries.append(text)
            lookup(search, text, f"search:{text}", fresh, round_number)

        if not fresh:
            log(f"  [{round_number}] no new candidates; stopping")
            break

        pool = list(fresh.values())
        if len(pool) > HYDRATE_CAP:
            pool.sort(key=lambda c: c.support, reverse=True)
        try:
            source.hydrate([c.paper for c in pool[:HYDRATE_CAP]])
        except Exception as e:  # noqa: BLE001 - judge on titles if abstracts can't be had
            log(f"  [{round_number}] ! abstract fetch failed: {e}")

        # Several routes reaching one paper (co-citation, graph + search) is the
        # signal that it belongs near the top of what the LLM is shown.
        pool.sort(key=lambda c: c.support, reverse=True)
        run.candidates.update(fresh)

        shortlist = pool[: config.budget]
        for candidate in shortlist:
            paper = candidate.paper
            text = f"{paper.title}. {paper.abstract}".strip()
            try:
                candidate.assessment = assess(llm, paper.title, text, topic[:1500])
            except Exception as e:  # noqa: BLE001 - leave this one unjudged
                log(f"  [{round_number}] ! assessment failed: {e}")
            mark = "?" if candidate.relevance is None else candidate.relevance
            log(f"  [{round_number}] [{mark}] {paper.title[:90]}")
        accepted = [
            candidate
            for candidate in shortlist
            if candidate.relevance is not None and candidate.relevance >= config.accept
        ]

        share = len(accepted) / len(shortlist)
        log(
            f"  [{round_number}] {len(pool)} new candidates, "
            f"{len(shortlist)} judged, {len(accepted)} accepted"
        )

        if share < config.min_yield:
            log(
                f"  [{round_number}] yield {share:.0%} is under {config.min_yield:.0%}; stopping"
            )
            break

        frontier = [c.paper for c in accepted[: config.expand]]
        queries = (
            generate_queries(
                llm,
                topic[:1500],
                [c.paper.title for c in accepted[:8]],
                run.queries,
                config.queries_per_round,
            )
            if round_number < config.rounds
            else []
        )

    return run
