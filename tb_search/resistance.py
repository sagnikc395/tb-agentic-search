"""Predict resistance for one isolate and drug, and check the reasoning trace.

A fluent trace is not a valid one. Every prediction is stored next to the
variants it was shown, so three things can be checked mechanically: that each
cited variant was really in the VCF, that each cited gene is really graded for
that drug, and that a "resistant" call actually rests on a variant. Those checks
are what separates a biologically grounded trace from a plausible-sounding one.

Two traces are kept per prediction, because they can disagree: the reasoning the
model wrote into its answer, and the hidden thinking the server separated out
(vLLM's `reasoning` field, when it runs with a reasoning parser). The second is
the one that shows how the call was actually reached, so it is stored verbatim
for the biological-validity read-through.
"""

from dataclasses import dataclass, field

from tb_search.catalogue import Panel, panel_for
from tb_search.cohort import Sample
from tb_search.prompts import RESISTANCE_PROMPT
from tb_search.screening import run_agent
from tb_search.vcf import Variant, VcfError, panel_gene_set, read_vcf
from tb_search.verdicts import _load_json, as_confidence

CALLS = ("resistant", "not_resistant", "unknown")
# What the model may say instead of the three call values above.
CALL_ALIASES = {
    "r": "resistant",
    "resistance": "resistant",
    "susceptible": "not_resistant",
    "sensitive": "not_resistant",
    "s": "not_resistant",
    "not resistant": "not_resistant",
    "non_resistant": "not_resistant",
    "no": "not_resistant",
    "yes": "resistant",
    "uncertain": "unknown",
    "indeterminate": "unknown",
    "unclear": "unknown",
}


def parse_prediction(answer: str) -> dict | None:
    """Pull the one {"call": ...} object out of the reply; None when none parses."""
    for data in _load_json(answer or ""):
        candidates = data if isinstance(data, list) else [data]
        for item in candidates:
            if isinstance(item, dict) and ("call" in item or "prediction" in item):
                return item
    return None


def normalize_call(value) -> str:
    """Coerce whatever the model said into one of the three calls; "unknown" if unsure."""
    folded = str(value or "").strip().lower().replace("-", "_")
    if folded in CALLS:
        return folded
    return CALL_ALIASES.get(
        folded, CALL_ALIASES.get(folded.replace("_", " "), "unknown")
    )


def _int_ids(value) -> list[int]:
    """The variant ids a reply cited, however loosely it wrote them."""
    if isinstance(value, (int, str)):
        value = [value]
    if not isinstance(value, list):
        return []
    ids = []
    for item in value:
        try:  # a model sometimes writes "0" or "variant 0"
            ids.append(int(str(item).strip().split()[-1]))
        except (ValueError, IndexError):
            continue
    return ids


def _strings(value) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def variant_listing(variants: list[Variant], panel: Panel) -> str:
    """The numbered variant block shown to the model; ids are list indices."""
    if not variants:
        return "(no variants called in any of the panel genes)"
    lines = []
    for index, variant in enumerate(variants):
        tier = panel.tier_of(variant.gene)
        marker = f"tier {tier}" if tier else "off-panel"
        lines.append(f"[{index}] {variant.label()}  — {marker}")
    return "\n".join(lines)


@dataclass
class TraceCheck:
    """Whether a reasoning trace holds up against the variants it was given."""

    # Every cited id pointed at a variant that was actually in the listing.
    variants_resolve: bool = True
    # Every cited gene is graded for this drug.
    genes_on_panel: bool = True
    # A "resistant" call cites at least one variant.
    call_grounded: bool = True
    # The call rests on a tier 1 gene rather than only a secondary one.
    tier1_support: bool = False
    # Whether the server returned separated thinking for this call, and how
    # much of it: an empty trace means there is nothing to read through, not
    # that the reasoning was sound.
    thinking_captured: bool = False
    thinking_words: int = 0
    hallucinated_ids: list[int] = field(default_factory=list)
    off_panel_genes: list[str] = field(default_factory=list)
    cited_labels: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        """A trace is valid when nothing in it was invented or self-contradictory."""
        return self.variants_resolve and self.genes_on_panel and self.call_grounded

    def as_dict(self) -> dict:
        return {
            "valid": self.valid,
            "variants_resolve": self.variants_resolve,
            "genes_on_panel": self.genes_on_panel,
            "call_grounded": self.call_grounded,
            "tier1_support": self.tier1_support,
            "thinking_captured": self.thinking_captured,
            "thinking_words": self.thinking_words,
            "hallucinated_ids": self.hallucinated_ids,
            "off_panel_genes": self.off_panel_genes,
            "cited_variants": self.cited_labels,
            "notes": self.notes,
        }


def check_trace(prediction: dict, variants: list[Variant], panel: Panel) -> TraceCheck:
    """Grade one trace against the variants the model was shown."""
    check = TraceCheck()
    cited_ids = prediction.get("variants_cited") or []
    resolved: list[Variant] = []
    for ident in cited_ids:
        if 0 <= ident < len(variants):
            resolved.append(variants[ident])
        else:
            check.hallucinated_ids.append(ident)
    if check.hallucinated_ids:
        check.variants_resolve = False
        check.notes.append(
            f"cited variant id(s) not in the listing: {check.hallucinated_ids}"
        )
    check.cited_labels = [variant.label() for variant in resolved]

    panel_names = {name.lower() for name in panel.genes}
    for gene in prediction.get("genes_cited") or []:
        if gene.lower() not in panel_names:
            check.off_panel_genes.append(gene)
    if check.off_panel_genes:
        check.genes_on_panel = False
        check.notes.append(
            f"gene(s) not graded for {panel.drug}: {', '.join(check.off_panel_genes)}"
        )

    if prediction.get("call") == "resistant" and not resolved:
        check.call_grounded = False
        check.notes.append("called resistant without citing any variant")

    check.tier1_support = any(panel.tier_of(v.gene) == 1 for v in resolved)

    thinking = str(prediction.get("reasoning_trace") or "")
    check.thinking_captured = bool(thinking.strip())
    check.thinking_words = len(thinking.split())
    if not check.thinking_captured:
        check.notes.append(
            "no separated reasoning trace returned; serve the model with a "
            "vLLM reasoning parser to capture one"
        )
    return check


def predict(
    agent, sample: Sample, variants: list[Variant], panel: Panel
) -> dict | None:
    """One A1 run for one isolate/drug; None when no reply could be parsed.

    The returned prediction carries `reasoning_trace`: the thinking the server
    separated from the answer, collected across every step of the run. It is
    empty when the model was served without a reasoning parser and wrote none.

    The prompt hands the agent the whole variant list and tells it not to go
    looking, so the expected run is a single <solution> step — but it is still
    an A1 run, and its tool loop stays available for a variant it wants to read
    up on.
    """
    prompt = (
        RESISTANCE_PROMPT.replace("{DRUG}", panel.drug)
        .replace("{TIER1}", ", ".join(panel.tier1) or "none")
        .replace("{TIER2}", ", ".join(panel.tier2) or "none")
        .replace("{ISOLATE}", sample.isolate)
        .replace("{COUNT}", str(len(variants)))
        .replace("{VARIANTS}", variant_listing(variants, panel))
    )
    if panel.is_fallback:
        prompt += (
            f"\n\nNOTE: no gene panel is graded for {panel.drug}. The variants "
            "shown span every TB resistance gene, so say so rather than forcing a "
            "mechanism; 'unknown' is the honest call unless the link is real."
        )

    for strict in (False, True):  # second pass only asks for stricter JSON
        if strict:
            prompt += "\n\nREMINDER: reply with ONLY the JSON object, no other text."
        answer, trace = run_agent(agent, prompt)
        data = parse_prediction(answer)
        if data:
            return normalize_prediction(data, trace)
    return None


def normalize_prediction(data: dict, reasoning_trace: str = "") -> dict:
    """Coerce a parsed reply into the one prediction shape."""
    call = normalize_call(data.get("call", data.get("prediction")))
    return {
        "call": call,
        "confidence": as_confidence(data.get("confidence")),
        "variants_cited": _int_ids(data.get("variants_cited")),
        "genes_cited": _strings(data.get("genes_cited")),
        "mechanism": str(data.get("mechanism", "")).strip(),
        "reasoning": str(data.get("reasoning", "")).strip(),
        # The server-separated thinking, kept verbatim: it is the trace whose
        # biological validity is being judged.
        "reasoning_trace": reasoning_trace.strip(),
        "unknown_reason": str(data.get("unknown_reason", "")).strip()
        if call == "unknown"
        else "",
    }


def predict_sample(
    agent,
    sample: Sample,
    panel: Panel,
    pass_only: bool = True,
    min_allele_frequency: float = 0.0,
) -> dict:
    """Read one isolate's VCF, predict, and grade the trace. Always returns a row.

    A missing or unreadable VCF yields an "unknown" row rather than raising, so
    one bad file cannot end a run over thousands of isolates.
    """
    row = {
        "isolate": sample.isolate,
        "drug": panel.drug,
        "phenotype": sample.phenotype,
        "vcf_path": sample.vcf_path,
    }
    try:
        read = read_vcf(
            sample.vcf_path,
            genes=panel_gene_set(panel),
            pass_only=pass_only,
            min_allele_frequency=min_allele_frequency,
        )
    except VcfError as e:
        return {
            **row,
            "call": "unknown",
            "confidence": None,
            "error": str(e),
            "variants": [],
            "reasoning": "VCF could not be read",
            "reasoning_trace": "",
            "trace": None,
        }

    variants = read.by_tier(panel)
    try:
        prediction = predict(agent, sample, variants, panel)
    except Exception as e:  # noqa: BLE001 - one failed isolate must not stop the run
        return {
            **row,
            "call": "unknown",
            "confidence": None,
            "error": f"prediction call failed: {e}",
            "variants": [variant.as_dict() for variant in variants],
            "reasoning": "",
            "reasoning_trace": "",
            "trace": None,
        }

    if prediction is None:
        return {
            **row,
            "call": "unknown",
            "confidence": None,
            "error": "no parseable prediction returned",
            "variants": [variant.as_dict() for variant in variants],
            "reasoning": "",
            "reasoning_trace": "",
            "trace": None,
        }

    check = check_trace(prediction, variants, panel)
    return {
        **row,
        **prediction,
        "error": "",
        "vcf_records": read.total_records,
        "vcf_annotated": read.annotated,
        "ref_mismatches": read.ref_mismatches,
        "variants": [variant.as_dict() for variant in variants],
        "trace": check.as_dict(),
    }


def predict_samples(
    agent,
    samples: list[Sample],
    drug: str,
    pass_only: bool = True,
    min_allele_frequency: float = 0.0,
    log=print,
) -> list[dict]:
    """Predict over a per-drug dataset, one agent run per isolate."""
    panel = panel_for(drug)
    rows: list[dict] = []
    for number, sample in enumerate(samples, start=1):
        log(f"  {number}/{len(samples)} {sample.isolate} ({sample.phenotype})...")
        row = predict_sample(agent, sample, panel, pass_only, min_allele_frequency)
        rows.append(row)
        log(f"    {prediction_line(row)}")
    return rows


def prediction_line(row: dict) -> str:
    """One console line: the call, whether it matched the DST, and the trace verdict."""
    label = {"resistant": "R", "not_resistant": "S", "unknown": "?"}.get(
        row.get("call", ""), "?"
    )
    truth = row.get("phenotype", "?")
    if label == "?":
        mark = "--"
    else:
        mark = "hit " if label == truth else "MISS"
    trace = row.get("trace") or {}
    flag = (
        ""
        if trace.get("valid", True)
        else "  [trace: " + "; ".join(trace.get("notes") or ["invalid"]) + "]"
    )
    reason = row.get("error") or row.get("reasoning") or "no reasoning given"
    return f"[{mark}] call={label} dst={truth}{flag} — {reason[:110]}"
