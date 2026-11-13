from tb_search.assess import normalize_assessment
from tb_search.documents import section_excerpt
from tb_search.verdicts import parse_assessment, parse_paper_list, parse_string_list


def test_assessment_is_found_inside_prose_and_fences():
    reply = 'Sure!\n```json\n{"relevance": 3, "tb_central": true}\n```\nHope that helps.'
    assert parse_assessment(reply) == {"relevance": 3, "tb_central": True}
    assert parse_assessment("no json here") is None


def test_string_list_accepts_a_wrapped_list():
    assert parse_string_list('{"queries": ["rpoB rifampicin", " ", 3]}') == ["rpoB rifampicin"]
    assert parse_string_list("nothing") == []


def test_paper_list_drops_untitled_rows():
    rows = parse_paper_list('[{"title": "A", "url": "http://x"}, {"year": "2020"}]')
    assert [row["title"] for row in rows] == ["A"]
    assert rows[0]["link"] == "http://x"


def test_evidence_quote_is_checked_against_the_text():
    text = "We predict rifampicin resistance in M. tuberculosis from rpoB sequence."
    quoted = {"relevance": "3", "tb_central": "yes", "evidence": "predict  rifampicin resistance in M. Tuberculosis"}
    invented = {"relevance": 3, "evidence": "We sequenced 10,000 isolates."}
    assert normalize_assessment(quoted, text)["evidence_verified"] is True
    assert normalize_assessment(quoted, text)["relevance"] == 3
    assert normalize_assessment(invented, text)["evidence_verified"] is False


def test_abstention_is_not_a_negative():
    assessment = normalize_assessment({"relevance": 0, "abstain": True, "tb_central": False}, "")
    assert assessment["relevance"] is None and assessment["tb_central"] is None
    # A reply with no usable score is an abstention too, not a zero.
    assert normalize_assessment({"relevance": "high"}, "")["abstain"] is True


def test_facets_are_always_lists_and_paper_type_is_closed():
    assessment = normalize_assessment({"relevance": 9, "drugs": "isoniazid", "paper_type": "Opinion"}, "")
    assert assessment["drugs"] == ["isoniazid"] and assessment["genes_variants"] == []
    assert assessment["relevance"] == 3 and assessment["paper_type"] == "other"


def test_section_excerpt_reaches_methods_and_conclusion():
    text = (
        "Title Abstract " + "intro " * 400
        + "Methods We sequenced isolates. " + "detail " * 400
        + "Discussion Our model finds rpoB. " + "more " * 400
    )
    excerpt = section_excerpt(text, 1000)
    assert excerpt.startswith("Title Abstract")
    assert "Methods We sequenced" in excerpt and "Discussion Our model" in excerpt
    assert len(excerpt) < 1100


def test_section_excerpt_falls_back_to_the_opening():
    text = "word " * 1000
    assert section_excerpt(text, 500) == text[:500]
    assert section_excerpt("short", 500) == "short"
