"""Europe PMC as the literature source: seed lookup, citation graph, keyword search.

Every Paper here comes out of an API record, so a retrieved row can't be an
invented title. Europe PMC is free and keyless, and it indexes bioRxiv/medRxiv
preprints along with their reference lists.
"""

import html
import re
import time
from dataclasses import dataclass

import requests

EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"

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


@dataclass
class Paper:
    """One literature record. key is Europe PMC's "<source>:<id>"."""

    key: str
    title: str
    year: str = ""
    doi: str = ""
    abstract: str = ""
    cited_by: int = 0

    @property
    def link(self) -> str:
        if self.doi:
            return f"https://doi.org/{self.doi}"
        source, _, ident = self.key.partition(":")
        return f"https://europepmc.org/article/{source}/{ident}"


def _clean(text: str) -> str:
    """Europe PMC titles and abstracts carry inline HTML (<i>, <h4>, ...), sometimes escaped."""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html.unescape(text or ""))).strip()


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
            " OR ".join(f'DOI:"{prefix}/{biorxiv.group(1)}"' for prefix in BIORXIV_PREFIXES)
        )
    else:
        doi = DOI.search(identifier)
        if doi:
            queries.append(f'DOI:"{doi.group(0).rstrip(".,;")}"')
    return queries


def _http_get(url: str, params: dict) -> dict:
    """GET returning JSON; a flaky network gets two more tries."""
    for attempt in range(3):
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError):
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
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

    def _linked(self, paper: Paper, edge: str, limit: int) -> list[Paper]:
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
        """Fill in abstract/doi in place — graph records come back without them."""
        bare = {paper.key: paper for paper in papers if not paper.abstract}
        keys = list(bare)
        for start in range(0, len(keys), HYDRATE_BATCH):
            batch = keys[start : start + HYDRATE_BATCH]
            query = " OR ".join(
                "(EXT_ID:{1} AND SRC:{0})".format(*key.split(":", 1)) for key in batch
            )
            for full in self.search(query, limit=len(batch)):
                paper = bare.get(full.key)
                if paper:
                    paper.abstract = full.abstract
                    paper.doi = paper.doi or full.doi
                    paper.cited_by = paper.cited_by or full.cited_by
