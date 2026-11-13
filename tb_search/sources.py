"""The literature sources: seed lookup, citation graph, keyword search.

Every Paper here comes out of an API record, so a retrieved row can't be an
invented title. Three sources sit behind the same four lookups:

- Europe PMC: free and keyless, indexes bioRxiv/medRxiv preprints along with
  their reference lists, but holds little outside the life sciences.
- Semantic Scholar: the citation graph across biology and computer science,
  so the ML papers a TB methods paper cites are reachable.
- Crossref: the reference lists publishers deposited, which fill in what the
  other two failed to match. It has no public cited-by lookup.

Combined asks all of them and merges the records by DOI.
"""

import html
import os
import re
import time
from dataclasses import dataclass
from functools import partial
from urllib.parse import quote

import requests
from dotenv import load_dotenv

EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
S2_API = "https://api.semanticscholar.org/graph/v1"
CROSSREF = "https://api.crossref.org/works"

# Key prefixes for records that did not come from Europe PMC.
S2, CR = "S2", "CR"
ARXIV_DOI = "10.48550/arxiv."
S2_FIELDS = "title,year,abstract,externalIds,citationCount"
CROSSREF_FIELDS = "DOI,title,abstract,issued,is-referenced-by-count"
# Crossref also registers figures, datasets and peer reviews; a search wants papers.
CROSSREF_TYPES = "type:journal-article,type:posted-content,type:proceedings-article"

DOI = re.compile(r"10\.\d{4,9}/[^\s\"'<>)\]]+")
PUBMED_URL = re.compile(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)")
ARXIV_URL = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})")
# bioRxiv/medRxiv DOIs share one suffix across the old (10.1101) and new
# (10.64898) prefixes, and their URLs carry a version tag the DOI doesn't.
BIORXIV_SUFFIX = re.compile(r"(\d{4}\.\d{2}\.\d{2}\.\d{6,8})(?:v\d+)?")
BIORXIV_PREFIXES = ("10.1101", "10.64898")

# Europe PMC caps a page at 1000 rows; ids are looked up a batch at a time to
# keep the query string short.
MAX_PAGE = 1000
HYDRATE_BATCH = 20
# Semantic Scholar caps a keyword search at 100 rows.
S2_SEARCH_MAX = 100


@dataclass
class Paper:
    """One literature record.

    key is Europe PMC's "<source>:<id>", or "S2:<paperId>" / "CR:<doi>" for a
    record only Semantic Scholar or Crossref returned.
    """

    key: str
    title: str
    year: str = ""
    doi: str = ""
    abstract: str = ""
    cited_by: int = 0

    @property
    def source(self) -> str:
        """The index this record's key belongs to."""
        return {S2: "semanticscholar", CR: "crossref"}.get(
            self.key.partition(":")[0], "europepmc"
        )

    @property
    def link(self) -> str:
        if self.doi:
            return f"https://doi.org/{self.doi}"
        source, _, ident = self.key.partition(":")
        if source == S2:
            return f"https://www.semanticscholar.org/paper/{ident}"
        return f"https://europepmc.org/article/{source}/{ident}"


def _clean(text: str) -> str:
    """Europe PMC titles and abstracts carry inline HTML (<i>, <h4>, ...), sometimes escaped."""
    return re.sub(
        r"\s+", " ", re.sub(r"<[^>]+>", " ", html.unescape(text or ""))
    ).strip()


def _paper(record: dict) -> Paper | None:
    """A Paper from a search/reference/citation record; None when it has no id."""
    source, ident = record.get("source"), record.get("id")
    title = _clean(record.get("title", ""))
    if not source or not ident or not title:
        return None  # an unmatched reference: free text only, nothing to verify
    year = record.get("pubYear") or ""
    return Paper(
        key=f"{source}:{ident}",
        title=title.rstrip(". "),
        year=str(year) if year else "",
        doi=(record.get("doi") or "").lower(),
        abstract=_clean(record.get("abstractText", "")),
        cited_by=int(record.get("citedByCount") or 0),
    )


def seed_queries(identifier: str) -> list[str]:
    """Europe PMC queries that would pin down the paper a URL/DOI/PMID names."""
    queries = []
    pubmed = PUBMED_URL.search(identifier)
    if pubmed:
        queries.append(f"(EXT_ID:{pubmed.group(1)} AND SRC:MED)")
    arxiv = ARXIV_URL.search(identifier)
    if arxiv:
        queries.append(f'DOI:"10.48550/arxiv.{arxiv.group(1)}"')
    biorxiv = BIORXIV_SUFFIX.search(identifier)
    if biorxiv:
        queries.append(
            " OR ".join(
                f'DOI:"{prefix}/{biorxiv.group(1)}"' for prefix in BIORXIV_PREFIXES
            )
        )
    else:
        doi = DOI.search(identifier)
        if doi:
            queries.append(f'DOI:"{doi.group(0).rstrip(".,;")}"')
    return queries


def seed_dois(identifier: str) -> list[str]:
    """The DOIs a URL/DOI could stand for, for the sources that look papers up by DOI."""
    dois = []
    arxiv = ARXIV_URL.search(identifier)
    if arxiv:
        dois.append(f"{ARXIV_DOI}{arxiv.group(1)}")
    biorxiv = BIORXIV_SUFFIX.search(identifier)
    if biorxiv:
        dois += [f"{prefix}/{biorxiv.group(1)}" for prefix in BIORXIV_PREFIXES]
    else:
        doi = DOI.search(identifier)
        if doi:
            dois.append(doi.group(0).rstrip(".,;").lower())
    return dois


def _plain(query: str) -> str:
    """Europe PMC's OR syntax means nothing to the other sources; they take bare keywords."""
    return re.sub(r"\s+OR\s+", " ", query).strip()


def _http_get(url: str, params: dict, headers: dict | None = None) -> dict:
    """GET returning JSON; a flaky network gets two more tries. A 404 is an empty answer."""
    for attempt in range(3):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)
            if response.status_code == 404:
                return {}  # the source doesn't hold this paper
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as e:
            if attempt == 2:
                raise
            # A rate limit (keyless Semantic Scholar, mostly) needs a longer pause.
            limited = getattr(getattr(e, "response", None), "status_code", None) == 429
            time.sleep((5 if limited else 2) * (attempt + 1))
    return {}


class EuropePMC:
    """The four lookups the discovery loop needs. `fetch` is swappable for tests."""

    def __init__(self, fetch=_http_get):
        self.fetch = fetch

    def search(self, query: str, limit: int = 25) -> list[Paper]:
        """Best matches first, abstracts included."""
        data = self.fetch(
            f"{EPMC}/search",
            {
                "query": query,
                "format": "json",
                "resultType": "core",
                "pageSize": min(max(1, limit), MAX_PAGE),
            },
        )
        records = data.get("resultList", {}).get("result", [])
        return [paper for paper in map(_paper, records) if paper]

    def resolve(self, identifier: str) -> Paper | None:
        """The paper a URL, DOI or PubMed link points at; None when it names none."""
        for query in seed_queries(identifier):
            found = self.search(query, limit=1)
            if found:
                return found[0]
        return None

    def _own(self, paper: Paper) -> Paper | None:
        """This paper as a Europe PMC record; one from another source is found by DOI."""
        if paper.source == "europepmc":
            return paper
        found = self.search(f'DOI:"{paper.doi}"', limit=1) if paper.doi else []
        return found[0] if found else None

    def _linked(self, paper: Paper, edge: str, limit: int) -> list[Paper]:
        paper = self._own(paper)
        if paper is None:
            return []
        source, _, ident = paper.key.partition(":")
        data = self.fetch(
            f"{EPMC}/{source}/{ident}/{edge}",
            {"format": "json", "pageSize": min(max(1, limit), MAX_PAGE)},
        )
        listing = data.get(f"{edge[:-1]}List", {}).get(edge[:-1], [])
        return [linked for linked in map(_paper, listing) if linked]

    def references(self, paper: Paper, limit: int = MAX_PAGE) -> list[Paper]:
        """Papers this one cites. References Europe PMC couldn't match are dropped."""
        return self._linked(paper, "references", limit)

    def citations(self, paper: Paper, limit: int = 100) -> list[Paper]:
        """Papers citing this one."""
        return self._linked(paper, "citations", limit)

    def hydrate(self, papers: list[Paper]) -> None:
        """Fill in abstract/doi in place — graph records come back without them.

        A record from another source is looked up by its DOI.
        """
        bare = [
            paper
            for paper in papers
            if not paper.abstract and (paper.source == "europepmc" or paper.doi)
        ]
        for start in range(0, len(bare), HYDRATE_BATCH):
            wanted: dict[str, Paper] = {}
            clauses = []
            for paper in bare[start : start + HYDRATE_BATCH]:
                if paper.source == "europepmc":
                    wanted[paper.key] = paper
                    clauses.append(
                        "(EXT_ID:{1} AND SRC:{0})".format(*paper.key.split(":", 1))
                    )
                else:
                    wanted[paper.doi] = paper
                    clauses.append(f'DOI:"{paper.doi}"')
            # A DOI can match a preprint and its published record, hence the headroom.
            for full in self.search(" OR ".join(clauses), limit=2 * len(clauses)):
                paper = wanted.get(full.key) or wanted.get(full.doi)
                if paper and not paper.abstract:
                    paper.abstract = full.abstract
                    paper.doi = paper.doi or full.doi
                    paper.cited_by = paper.cited_by or full.cited_by


class SemanticScholar:
    """The same lookups on Semantic Scholar. Keyless works, but an API key lifts the rate limit."""

    def __init__(self, fetch=_http_get, api_key: str = ""):
        self.fetch = (
            partial(fetch, headers={"x-api-key": api_key}) if api_key else fetch
        )

    @staticmethod
    def _paper(record: dict | None) -> Paper | None:
        record = record or {}
        ident, title = record.get("paperId"), _clean(record.get("title") or "")
        if not ident or not title:
            return None  # a reference Semantic Scholar couldn't match
        external = record.get("externalIds") or {}
        doi = (external.get("DOI") or "").lower()
        if not doi and external.get("ArXiv"):
            doi = f"{ARXIV_DOI}{external['ArXiv']}".lower()
        year = record.get("year")
        return Paper(
            key=f"{S2}:{ident}",
            title=title.rstrip(". "),
            year=str(year) if year else "",
            doi=doi,
            abstract=_clean(record.get("abstract") or ""),
            cited_by=int(record.get("citationCount") or 0),
        )

    @staticmethod
    def _ident(paper: Paper) -> str | None:
        """How Semantic Scholar addresses this paper; None when it has no id it knows."""
        source, _, ident = paper.key.partition(":")
        if source == S2:
            return ident
        if paper.doi.startswith(ARXIV_DOI):
            return f"ARXIV:{paper.doi[len(ARXIV_DOI) :]}"
        if paper.doi:
            return f"DOI:{quote(paper.doi, safe='/')}"
        return {"MED": f"PMID:{ident}", "PMC": f"PMCID:{ident}"}.get(source)

    def search(self, query: str, limit: int = 25) -> list[Paper]:
        data = self.fetch(
            f"{S2_API}/paper/search",
            {
                "query": _plain(query),
                "fields": S2_FIELDS,
                "limit": min(max(1, limit), S2_SEARCH_MAX),
            },
        )
        return [paper for paper in map(self._paper, data.get("data") or []) if paper]

    def resolve(self, identifier: str) -> Paper | None:
        idents = [
            f"PMID:{match.group(1)}"
            for match in [PUBMED_URL.search(identifier)]
            if match
        ]
        idents += [
            self._ident(Paper(key="", title="", doi=doi))
            for doi in seed_dois(identifier)
        ]
        for ident in idents:
            paper = self._paper(
                self.fetch(f"{S2_API}/paper/{ident}", {"fields": S2_FIELDS})
            )
            if paper:
                return paper
        return None

    def _linked(self, paper: Paper, edge: str, end: str, limit: int) -> list[Paper]:
        ident = self._ident(paper)
        if ident is None:
            return []
        data = self.fetch(
            f"{S2_API}/paper/{ident}/{edge}",
            {"fields": S2_FIELDS, "limit": min(max(1, limit), MAX_PAGE)},
        )
        rows = data.get("data") or []
        return [
            linked for linked in (self._paper(row.get(end)) for row in rows) if linked
        ]

    def references(self, paper: Paper, limit: int = MAX_PAGE) -> list[Paper]:
        return self._linked(paper, "references", "citedPaper", limit)

    def citations(self, paper: Paper, limit: int = 100) -> list[Paper]:
        return self._linked(paper, "citations", "citingPaper", limit)

    def hydrate(self, papers: list[Paper]) -> None:
        """Nothing to do: graph records already carry whatever abstract there is."""


class Crossref:
    """Crossref: DOI lookup, deposited reference lists, keyword search. No cited-by lookup.

    `mailto` puts requests in Crossref's faster "polite" pool.
    """

    def __init__(self, fetch=_http_get, mailto: str = ""):
        self.fetch = fetch
        self.base = {"mailto": mailto} if mailto else {}

    @staticmethod
    def _paper(record: dict) -> Paper | None:
        doi = (record.get("DOI") or "").lower()
        title = _clean(" ".join((record.get("title") or [])[:1]))
        if not doi or not title:
            return None
        dates = (record.get("issued") or {}).get("date-parts") or [[]]
        year = dates[0][0] if dates[0] else None
        return Paper(
            key=f"{CR}:{doi}",
            title=title.rstrip(". "),
            year=str(year) if year else "",
            doi=doi,
            abstract=_clean(record.get("abstract") or ""),
            cited_by=int(record.get("is-referenced-by-count") or 0),
        )

    def _items(self, params: dict) -> list[Paper]:
        data = self.fetch(CROSSREF, {**self.base, "select": CROSSREF_FIELDS, **params})
        records = data.get("message", {}).get("items") or []
        return [paper for paper in map(self._paper, records) if paper]

    def _work(self, doi: str) -> dict:
        return self.fetch(f"{CROSSREF}/{quote(doi, safe='/')}", dict(self.base)).get(
            "message", {}
        )

    def search(self, query: str, limit: int = 25) -> list[Paper]:
        return self._items(
            {
                "query.bibliographic": _plain(query),
                "filter": CROSSREF_TYPES,
                "rows": min(max(1, limit), MAX_PAGE),
            }
        )

    def resolve(self, identifier: str) -> Paper | None:
        for doi in seed_dois(identifier):
            paper = self._paper(self._work(doi))
            if paper:
                return paper
        return None

    def references(self, paper: Paper, limit: int = MAX_PAGE) -> list[Paper]:
        """Papers this one cites. A deposited reference without a DOI is dropped."""
        if not paper.doi:
            return []
        dois: dict[str, None] = {}
        for reference in self._work(paper.doi).get("reference") or []:
            if reference.get("DOI"):
                dois.setdefault(reference["DOI"].lower())
        wanted, found = list(dois)[:limit], []
        # The list carries DOIs but rarely titles, so the records are fetched a batch at a time.
        for start in range(0, len(wanted), HYDRATE_BATCH):
            batch = wanted[start : start + HYDRATE_BATCH]
            found += self._items(
                {"filter": ",".join(f"doi:{doi}" for doi in batch), "rows": len(batch)}
            )
        return found

    def citations(self, paper: Paper, limit: int = 100) -> list[Paper]:
        return []

    def hydrate(self, papers: list[Paper]) -> None:
        """Nothing to do: a Crossref record arrives with its abstract when one was deposited."""


def _title_key(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


class Combined:
    """Several sources behind the same four lookups, their records merged into one Paper each.

    Two records are the same paper when they share a DOI, or failing that a
    title. The first source to return a paper names its key, so list the
    preferred source first. A source that fails is skipped; the lookup only
    raises when every source failed.
    """

    def __init__(self, backends: list):
        self.backends = backends
        self._by_doi: dict[str, Paper] = {}
        self._by_title: dict[str, Paper] = {}

    def _merge(self, paper: Paper) -> Paper:
        """The one Paper kept for this record, topped up with what this copy adds."""
        title = _title_key(paper.title)
        kept = self._by_doi.get(paper.doi) if paper.doi else None
        kept = kept or self._by_title.get(title)
        if kept is None:
            kept = paper
        elif kept is not paper:
            kept.abstract = kept.abstract or paper.abstract
            kept.doi = kept.doi or paper.doi
            kept.year = kept.year or paper.year
            kept.cited_by = max(kept.cited_by, paper.cited_by)
        if kept.doi:
            self._by_doi.setdefault(kept.doi, kept)
        self._by_title.setdefault(title, kept)
        return kept

    def _gather(self, lookup: str, *args) -> list[Paper]:
        merged: dict[str, Paper] = {}
        failures = []
        for backend in self.backends:
            try:
                papers = getattr(backend, lookup)(*args)
            except Exception as e:  # noqa: BLE001 - the other sources can still answer
                failures.append(e)
                continue
            for paper in papers:
                kept = self._merge(paper)
                merged.setdefault(kept.key, kept)
        if failures and len(failures) == len(self.backends):
            raise failures[-1]
        return list(merged.values())

    def search(self, query: str, limit: int = 25) -> list[Paper]:
        """Up to `limit` from each source, so the merged list can be longer."""
        return self._gather("search", query, limit)

    def resolve(self, identifier: str) -> Paper | None:
        failures = []
        for backend in self.backends:
            try:
                paper = backend.resolve(identifier)
            except Exception as e:  # noqa: BLE001 - try the next source
                failures.append(e)
                continue
            if paper:
                return self._merge(paper)
        if failures and len(failures) == len(self.backends):
            raise failures[-1]
        return None

    def references(self, paper: Paper, limit: int = MAX_PAGE) -> list[Paper]:
        return self._gather("references", paper, limit)

    def citations(self, paper: Paper, limit: int = 100) -> list[Paper]:
        return self._gather("citations", paper, limit)

    def hydrate(self, papers: list[Paper]) -> None:
        for backend in self.backends:
            backend.hydrate(papers)


SOURCE_NAMES = ("europepmc", "semanticscholar", "crossref")


def build_source(names: list[str] | None = None):
    """The literature source for a run: one index, or Combined over several.

    SEMANTIC_SCHOLAR_API_KEY and CROSSREF_MAILTO are optional; both sources
    answer without them, only slower.
    """
    load_dotenv()
    makers = {
        "europepmc": EuropePMC,
        "semanticscholar": lambda: SemanticScholar(
            api_key=os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
        ),
        "crossref": lambda: Crossref(mailto=os.getenv("CROSSREF_MAILTO", "").strip()),
    }
    unknown = [name for name in names or [] if name not in makers]
    if unknown:
        raise ValueError(
            f"unknown source(s): {', '.join(unknown)}; pick from {SOURCE_NAMES}"
        )
    backends = [makers[name]() for name in names or SOURCE_NAMES]
    return backends[0] if len(backends) == 1 else Combined(backends)
