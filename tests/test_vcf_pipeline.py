"""Tests for the VCF side: gene lookup, variant parsing, the cohort join, trace checks.

Everything here is offline. The canonical resistance mutations are the real
assertion: if the H37Rv gene table or the codon arithmetic drifts, katG S315T
stops landing on codon 315 and these fail.
"""

import pytest

from tb_search.catalogue import drug_key, panel_for
from tb_search.cohort import Sample, load_cohort, remap_path
from tb_search.evaluate import evaluate_resistance
from tb_search.genome import consequence, genome_index, reference_sequence
from tb_search.resistance import (
    check_trace,
    normalize_call,
    normalize_prediction,
    parse_prediction,
    variant_listing,
)
from tb_search.vcf import VcfError, panel_gene_set, read_vcf

VCF_HEADER = (
    "##fileformat=VCFv4.2\n"
    "##reference=NC_000962.3\n"
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample\n"
)

# Canonical TB resistance mutations, with the label the pipeline should produce.
CANONICAL = [
    (761155, "C", "T", "rpoB codon 450"),
    (2155168, "C", "G", "katG codon 315"),
    (1673425, "C", "T", "fabG1 -15 (promoter)"),
    (1674481, "A", "G", "inhA codon 94"),
    (4247429, "A", "G", "embB codon 306"),
    (2289073, "G", "C", "pncA codon 57"),
    (7582, "A", "G", "gyrA codon 94"),
    (781687, "A", "G", "rpsL codon 43"),
    (1473246, "A", "G", "rrs 1401"),
]


def write_vcf(path, records, filter_value="PASS", genotype="1/1"):
    lines = [
        f"NC_000962.3\t{pos}\t.\t{ref}\t{alt}\t900\t{filter_value}\tDP=50\t"
        f"GT:DP:AD\t{genotype}:50:0,50"
        for pos, ref, alt in records
    ]
    path.write_text(VCF_HEADER + "\n".join(lines) + "\n")
    return path


@pytest.mark.parametrize(("pos", "ref", "alt", "label"), CANONICAL)
def test_canonical_mutations_land_on_the_right_codon(pos, ref, alt, label):
    assert genome_index().locate(pos).label() == label


def test_promoter_picks_the_nearest_start_codon():
    """fabG1 -15 sits inside a neighbouring gene's promoter window too."""
    locus = genome_index().locate(1673425)
    assert locus.gene.name == "fabG1"
    assert locus.region == "promoter"
    assert locus.offset == -15


def test_rna_genes_get_no_codon_number():
    """rrs is a drug target but has no reading frame."""
    locus = genome_index().locate(1473246)
    assert locus.gene.name == "rrs"
    assert locus.codon is None
    assert locus.offset == 1401


def test_reverse_strand_codons_count_from_the_far_end():
    katG = genome_index().gene("katG")
    assert katG.strand == "-"
    assert katG.codon_of(2155168) == 315


def test_reference_loads_and_is_the_right_length():
    assert len(reference_sequence()) == 4411532


@pytest.mark.parametrize(
    ("pos", "ref", "alt", "change", "effect"),
    [
        # katG and pncA are on the reverse strand, so REF is the forward base.
        (761155, "C", "T", "S450L", "missense"),
        (2155168, "C", "G", "S315T", "missense"),
        (4247429, "A", "G", "M306V", "missense"),
        (7582, "A", "G", "D94G", "missense"),
        (2289073, "G", "C", "H57D", "missense"),
        (781687, "A", "G", "K43R", "missense"),
        (1673425, "C", "T", "", "upstream"),
        (1473246, "A", "G", "", "non_coding"),
        (2289072, "CA", "C", "", "frameshift"),
    ],
)
def test_protein_consequence_against_the_reference(pos, ref, alt, change, effect):
    locus = genome_index().locate(pos)
    assert consequence(locus, pos, ref, alt) == (change, effect)


def test_synonymous_changes_are_named_as_such(tmp_path):
    """rpoB codon 450 TCG: the third base is silent, the second is S450L."""
    path = write_vcf(tmp_path / "syn.vcf", [(761156, "G", "A")])
    variant = read_vcf(path).variants[0]
    assert variant.effect == "synonymous"
    assert variant.synonymous is True
    assert variant.protein_change == "S450S"
    assert "synonymous" in variant.label()


def test_ref_mismatch_is_counted(tmp_path):
    """A REF that disagrees with H37Rv means the wrong reference or coordinates."""
    path = write_vcf(tmp_path / "bad.vcf", [(761155, "A", "T")])  # reference is C
    read = read_vcf(path)
    assert read.ref_mismatches == 1
    assert read.variants[0].ref_matches is False

    good = read_vcf(write_vcf(tmp_path / "good.vcf", [(761155, "C", "T")]))
    assert good.ref_mismatches == 0
    assert good.variants[0].ref_matches is True


def test_read_vcf_names_variants_without_annotation(tmp_path):
    path = write_vcf(tmp_path / "iso.vcf", [(761155, "C", "T")])
    read = read_vcf(path)
    assert read.annotated is False
    assert len(read.variants) == 1
    assert read.variants[0].gene == "rpoB"
    assert read.variants[0].locus.codon == 450
    assert read.variants[0].key() == "761155C>T"


def test_read_vcf_uses_the_annotation_when_there_is_one(tmp_path):
    path = tmp_path / "annotated.vcf"
    path.write_text(
        "##fileformat=VCFv4.2\n"
        '##INFO=<ID=ANN,Number=.,Type=String,Description="Functional annotations">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "NC_000962.3\t761155\t.\tC\tT\t900\tPASS\t"
        "ANN=T|missense_variant|MODERATE|rpoB|Rv0667|transcript|p.Ser450Leu\n"
    )
    read = read_vcf(path)
    assert read.annotated is True
    variant = read.variants[0]
    assert variant.protein_change == "S450L"
    assert variant.effect == "missense_variant"
    assert variant.label().startswith("rpoB S450L")


def test_non_pass_and_reference_calls_are_dropped(tmp_path):
    path = write_vcf(tmp_path / "low.vcf", [(761155, "C", "T")], filter_value="LowCov")
    assert read_vcf(path).variants == []
    assert read_vcf(path, pass_only=False).variants != []

    path = write_vcf(tmp_path / "ref.vcf", [(761155, "C", "T")], genotype="0/0")
    assert read_vcf(path).variants == []


def test_frameshift_is_flagged_but_in_frame_indels_are_not(tmp_path):
    # A 1 bp deletion shifts the reading frame; a 3 bp one does not.
    path = write_vcf(
        tmp_path / "indel.vcf", [(2289072, "CA", "C"), (2289070, "CAAA", "C")]
    )
    shift, in_frame = read_vcf(path).variants
    assert shift.frameshift is True and shift.kind == "deletion"
    assert in_frame.frameshift is False and in_frame.kind == "deletion"


def test_panel_filtering_keeps_only_the_drug_s_genes(tmp_path):
    path = write_vcf(
        tmp_path / "mixed.vcf",
        [(761155, "C", "T"), (2155168, "C", "G"), (1234567, "G", "A")],
    )
    rif = read_vcf(path, genes=panel_gene_set(panel_for("rifampicin")))
    assert [v.gene for v in rif.variants] == ["rpoB"]
    inh = read_vcf(path, genes=panel_gene_set(panel_for("isoniazid")))
    assert [v.gene for v in inh.variants] == ["katG"]


def test_allele_frequency_threshold(tmp_path):
    path = tmp_path / "het.vcf"
    path.write_text(
        VCF_HEADER
        + "NC_000962.3\t761155\t.\tC\tT\t900\tPASS\tDP=50\tGT:DP:AD\t0/1:50:40,10\n"
    )
    assert read_vcf(path).variants[0].allele_frequency == pytest.approx(0.2)
    assert read_vcf(path, min_allele_frequency=0.5).variants == []


def test_missing_vcf_raises(tmp_path):
    with pytest.raises(VcfError):
        read_vcf(tmp_path / "nope.vcf")


def test_drug_key_folds_the_table_s_spellings():
    assert drug_key("RIFAMPICIN") == "rifampicin"
    assert drug_key("PARA-AMINOSALICYLIC_ACID") == "para_aminosalicylic_acid"
    assert drug_key("PARA_AMINOSALICYLIC_ACID") == "para_aminosalicylic_acid"
    assert drug_key("ETHIOMIDE") == drug_key("ETHIONAMIDE") == "ethionamide"


def test_panel_tiers():
    panel = panel_for("isoniazid")
    assert panel.tier_of("katG") == 1
    assert panel.tier_of("ndh") == 2
    assert panel.tier_of("rpoB") is None
    assert panel.is_fallback is False


def test_ungraded_drug_falls_back_to_the_union_panel():
    panel = panel_for("imipenem")
    assert panel.is_fallback is True
    assert "rpoB" in panel.genes and "katG" in panel.genes


def test_cohort_joins_every_listed_vcf():
    """New_ID is the join key; every BIG-TB VCF should match a table row."""
    cohort = load_cohort()
    assert cohort.unmatched_vcfs == 0
    assert len(cohort.isolates) == 17942
    assert cohort.counts("rifampicin")["R"] > 0


def test_dataset_balances_and_is_reproducible():
    cohort = load_cohort()
    first = cohort.dataset("isoniazid", limit=10, balance=True, seed=3)
    second = cohort.dataset("isoniazid", limit=10, balance=True, seed=3)
    assert [s.isolate for s in first] == [s.isolate for s in second]
    assert sum(1 for s in first if s.resistant) == 5
    assert len(first) == 10


def test_vcf_root_remaps_the_cluster_path():
    cluster = "/project/pi_annagreen_umass_edu/x/SAMEA123_variants.vcf"
    assert remap_path(cluster, "/mnt/vcfs") == "/mnt/vcfs/SAMEA123_variants.vcf"
    assert remap_path(cluster, None) == cluster


def variants_for(tmp_path, records, drug):
    path = write_vcf(tmp_path / "t.vcf", records)
    panel = panel_for(drug)
    return read_vcf(path, genes=panel_gene_set(panel)).by_tier(panel), panel


def test_trace_check_passes_a_grounded_call(tmp_path):
    variants, panel = variants_for(tmp_path, [(2155168, "C", "G")], "isoniazid")
    check = check_trace(
        normalize_prediction(
            {"call": "resistant", "variants_cited": [0], "genes_cited": ["katG"]}
        ),
        variants,
        panel,
    )
    assert check.valid and check.tier1_support
    assert check.cited_labels == ["katG S315T (2155168 C>G)"]


def test_trace_check_catches_a_cited_variant_that_was_never_shown(tmp_path):
    variants, panel = variants_for(tmp_path, [(2155168, "C", "G")], "isoniazid")
    check = check_trace(
        normalize_prediction({"call": "resistant", "variants_cited": [0, 7]}),
        variants,
        panel,
    )
    assert not check.valid
    assert check.hallucinated_ids == [7]


def test_trace_check_catches_a_gene_not_graded_for_the_drug(tmp_path):
    variants, panel = variants_for(tmp_path, [(2155168, "C", "G")], "isoniazid")
    check = check_trace(
        normalize_prediction(
            {"call": "resistant", "variants_cited": [0], "genes_cited": ["rpoB"]}
        ),
        variants,
        panel,
    )
    assert not check.valid
    assert check.off_panel_genes == ["rpoB"]


def test_trace_check_catches_a_resistant_call_resting_on_nothing(tmp_path):
    variants, panel = variants_for(tmp_path, [], "isoniazid")
    check = check_trace(
        normalize_prediction({"call": "resistant", "variants_cited": []}),
        variants,
        panel,
    )
    assert not check.valid and not check.call_grounded


def test_a_susceptible_call_may_cite_nothing(tmp_path):
    """An absence of panel variants is legitimate grounds for calling susceptible."""
    variants, panel = variants_for(tmp_path, [], "isoniazid")
    check = check_trace(
        normalize_prediction({"call": "not_resistant", "variants_cited": []}),
        variants,
        panel,
    )
    assert check.valid


def test_variant_listing_numbers_and_tiers_the_variants(tmp_path):
    variants, panel = variants_for(
        tmp_path, [(2155168, "C", "G"), (2102000, "C", "T")], "isoniazid"
    )
    listing = variant_listing(variants, panel)
    assert listing.startswith("[0] katG S315T")
    assert "tier 1" in listing and "tier 2" in listing
    assert variant_listing([], panel).startswith("(no variants")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("resistant", "resistant"),
        ("Resistant", "resistant"),
        ("R", "resistant"),
        ("susceptible", "not_resistant"),
        ("sensitive", "not_resistant"),
        ("not-resistant", "not_resistant"),
        ("uncertain", "unknown"),
        ("something else", "unknown"),
    ],
)
def test_normalize_call(raw, expected):
    assert normalize_call(raw) == expected


def test_parse_prediction_survives_prose_and_fences():
    assert parse_prediction('{"call": "resistant"}')["call"] == "resistant"
    fenced = 'Sure:\n```json\n{"call": "susceptible", "genes_cited": ["katG"]}\n```\n'
    assert normalize_prediction(parse_prediction(fenced))["call"] == "not_resistant"
    assert parse_prediction("no json here") is None


def test_string_variant_ids_are_coerced():
    assert normalize_prediction({"call": "R", "variants_cited": ["0", "2"]})[
        "variants_cited"
    ] == [0, 2]


def test_unknown_reason_is_kept_only_for_unknown_calls():
    assert (
        normalize_prediction({"call": "resistant", "unknown_reason": "x"})[
            "unknown_reason"
        ]
        == ""
    )
    assert (
        normalize_prediction({"call": "unknown", "unknown_reason": "thin"})[
            "unknown_reason"
        ]
        == "thin"
    )


def test_evaluate_resistance_scores_calls_and_traces():
    def row(phenotype, call, valid=True, cited=("x",)):
        return {
            "isolate": "i",
            "phenotype": phenotype,
            "call": call,
            "reasoning": "",
            "trace": {
                "valid": valid,
                "variants_resolve": valid,
                "genes_on_panel": True,
                "call_grounded": True,
                "tier1_support": True,
                "cited_variants": list(cited),
            },
        }

    scores = evaluate_resistance(
        [
            row("R", "resistant"),
            row("R", "resistant"),
            row("R", "not_resistant"),
            row("S", "not_resistant"),
            row("S", "resistant", valid=False),
            row("S", "unknown"),
        ]
    )
    assert scores["n"] == 6
    assert scores["answered"] == 5
    assert scores["unknown"] == 1
    assert scores["unknown_by_phenotype"] == {"R": 0, "S": 1}
    assert scores["accuracy"] == 0.6
    assert scores["sensitivity"] == round(2 / 3, 3)
    assert scores["specificity"] == 0.5
    assert scores["trace"]["valid"] == round(5 / 6, 3)
    assert [m["isolate"] for m in scores["misses"]] == ["i", "i"]


def test_evaluate_resistance_handles_an_empty_run():
    scores = evaluate_resistance([])
    assert scores["accuracy"] is None
    assert scores["sensitivity"] is None
    assert scores["trace"]["valid"] is None


def test_error_rows_carry_no_trace_grade():
    """A VCF that could not be read must not count as a valid trace."""
    scores = evaluate_resistance(
        [
            {
                "isolate": "i",
                "phenotype": "R",
                "call": "unknown",
                "error": "gone",
                "trace": None,
            }
        ]
    )
    assert scores["errors"] == 1
    assert scores["trace"]["graded"] == 0
    assert scores["trace"]["valid"] is None


def test_sample_resistant_property():
    sample = Sample(isolate="i", drug="isoniazid", phenotype="R", vcf_path="/x.vcf")
    assert sample.resistant is True
