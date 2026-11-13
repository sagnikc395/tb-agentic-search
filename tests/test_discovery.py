import json

from tb_search.discovery import discover, keyword_query, lexical_scores
from tb_search.evaluate import evaluate_retrieval, evaluate_screening, recall_at
from tb_search.sources import EuropePMC, Paper, seed_queries
from tb_search.store import Store


def paper(key, title, abstract="abstract"):
    return Paper(key=key, title=title, abstract=abstract)


SEED = paper("PPR:1", "Benchmark for tuberculosis resistance prediction", "rifampicin rpoB genomes")
TB_REF = paper("MED:2", "Mycobacterium tuberculosis rifampicin resistance genomes")
OTHER_REF = paper("MED:3", "Protein language model pretraining")
CITER = paper("MED:4", "Deep learning for tuberculosis resistance prediction from genomes")
SECOND_HOP = paper("MED:5", "Tuberculosis rpoB resistance variant catalogue")
OFF_TOPIC = paper("MED:6", "Melon genome annotation")


class FakeSource:
    """A five-paper literature: the seed cites two papers, CITER cites two as well."""

    def __init__(self):
        self.references_of = {"PPR:1": [TB_REF, OTHER_REF], "MED:4": [TB_REF, SECOND_HOP]}
        self.citations_of = {"PPR:1": [CITER]}

    def resolve(self, identifier):
        return SEED if "10.1101/seed" in identifier else None

    def references(self, paper, limit=1000):
        return self.references_of.get(paper.key, [])

    def citations(self, paper, limit=100):
        return self.citations_of.get(paper.key, [])

    def search(self, query, limit=25):
        return [CITER, OFF_TOPIC]

    def hydrate(self, papers):
        pass


class FakeLLM:
    """Scores papers by whether the document part of the prompt mentions tuberculosis."""

    def __init__(self):
        self.assessed = 0

    def invoke(self, prompt):
        if "Write " in prompt and "search queries" in prompt:
            return '["rpoB variant catalogue"]'
        self.assessed += 1
        document = prompt.split("DOCUMENT", 1)[1].lower()
        return json.dumps({"relevance": 3 if "tubercul" in document else 0, "tb_central": True})


def quiet(*_args):
    pass


def test_seed_queries_cover_url_shapes():
    biorxiv = seed_queries("like https://www.biorxiv.org/content/10.1101/2026.01.30.702134v1")
    assert biorxiv == ['DOI:"10.1101/2026.01.30.702134" OR DOI:"10.64898/2026.01.30.702134"']
    assert seed_queries("https://pubmed.ncbi.nlm.nih.gov/35780211/") == ["(EXT_ID:35780211 AND SRC:MED)"]
    assert seed_queries("doi 10.1038/s41467-022-31236-0.") == ['DOI:"10.1038/s41467-022-31236-0"']
    assert seed_queries("https://arxiv.org/abs/2405.17631v3") == ['DOI:"10.48550/arxiv.2405.17631"']
    assert seed_queries("papers about TB drug resistance") == []


def test_europepmc_drops_unmatched_references_and_cleans_titles():
    def fetch(url, params):
        assert url.endswith("/PPR/1/references")
        return {
            "referenceList": {
                "reference": [
                    {"source": "MED", "id": "2", "title": "On &lt;i&gt;M. tuberculosis&lt;/i&gt;.", "pubYear": 2020},
                    {"unstructuredInformation": "Some et al. A book.", "match": "N"},
                ]
            }
        }

    (found,) = EuropePMC(fetch).references(SEED)
    assert (found.key, found.title, found.year) == ("MED:2", "On M. tuberculosis", "2020")
    assert found.link == "https://europepmc.org/article/MED/2"


def test_lexical_scores_prefer_topic_overlap():
    scores = lexical_scores(SEED.title, [TB_REF, CITER, OFF_TOPIC])
    assert scores["MED:4"] > scores["MED:2"] > scores["MED:6"] == 0
    assert keyword_query("Find the papers on rpoB and rifampicin") == "rpob OR rifampicin"


def test_discover_follows_accepted_papers_into_a_second_round():
    llm, store = FakeLLM(), Store()
    run = discover(FakeSource(), store, "like 10.1101/seed", llm, "fake", log=quiet)
    found = [c.paper.key for c in run.papers(10, 2)]
    # SECOND_HOP is reachable only through CITER, which round 1 accepted.
    assert set(found) == {"MED:2", "MED:4", "MED:5"}
    assert run.candidates["MED:6"].relevance == 0
    assert "PPR:1" not in run.candidates
    kinds = [kind for _round, kind, _detail in store.events(1)]
    assert kinds[0] == "seed" and "round" in kinds


def test_assessments_are_cached_across_runs():
    llm, store = FakeLLM(), Store()
    discover(FakeSource(), store, "like 10.1101/seed", llm, "fake", log=quiet)
    first = llm.assessed
    discover(FakeSource(), store, "like 10.1101/seed", llm, "fake", log=quiet)
    assert first > 0 and llm.assessed == first


def test_discover_without_llm_ranks_lexically():
    run = discover(FakeSource(), Store(), "like 10.1101/seed", log=quiet)
    ranked = [c.paper.key for c in run.papers(2, 2)]
    assert ranked[0] == "MED:4" and all(c.assessment is None for c in run.candidates.values())


def test_unresolvable_topic_with_no_hits_returns_nothing():
    class Empty(FakeSource):
        def search(self, query, limit=25):
            return []

    run = discover(Empty(), Store(), "papers about nothing at all", FakeLLM(), "fake", log=quiet)
    assert run.seed is None and run.papers(10, 2) == []


def test_retrieval_eval_holds_out_the_seed_references():
    report = evaluate_retrieval(FakeSource(), Store(), "like 10.1101/seed", log=quiet)
    assert report["gold"] == {"all": 2, "tb": 1}
    # TB_REF comes back through CITER's references; OTHER_REF is only in the hidden list.
    assert report["recall"]["tb"]["pool"] == 1.0
    assert report["recall"]["all"]["pool"] == 0.5
    assert evaluate_retrieval(FakeSource(), Store(), "no seed here", log=quiet) is None


def test_recall_at_cutoffs():
    assert recall_at(["a", "b", "c"], {"a", "c"}, cutoffs=(1, 3)) == {"@1": 0.5, "@3": 1.0, "pool": 1.0}
    assert recall_at(["a"], set(), cutoffs=(1,)) == {"@1": None, "pool": None}


def test_screening_eval_counts_abstentions_apart_from_errors(tmp_path):
    gold = tmp_path / "gold.json"
    gold.write_text('{"a.pdf": true, "b.pdf": false, "c.pdf": true}')
    rows = [
        {"file": "a.pdf", "tb_related": True, "evidence_verified": True},
        {"file": "b.pdf", "tb_related": True, "evidence_verified": False},
        {"file": "c.pdf", "tb_related": None},
        {"file": "unlabelled.pdf", "tb_related": True},
    ]
    scores = evaluate_screening(rows, gold)
    assert scores["accuracy"] == 0.5 and scores["wrong"] == ["b.pdf"]
    assert scores["abstained"] == ["c.pdf"] and scores["unverified_evidence"] == ["b.pdf"]
