"""Read one isolate's VCF into named, panel-filtered variants the agent can reason over.

Nothing here assumes the VCF was annotated. When an ANN/CSQ field is present its
gene and protein change are kept; otherwise every variant is named by looking
its H37Rv position up in the gene table, which is what the BIG-TB calls need.
"""

import gzip
import io
import re
from dataclasses import dataclass, field
from pathlib import Path

from tb_search.catalogue import Panel
from tb_search.genome import Locus, consequence, genome_index, reference_sequence

# Filter values that mean "the caller was happy with this call".
PASS_VALUES = frozenset({"PASS", ".", ""})
# Genotypes that mean the sample carries no alternate allele.
REF_GENOTYPES = frozenset({"0/0", "0|0", "0", "./.", ".|.", "."})

AMINO_ACIDS = {
    "Ala": "A",
    "Arg": "R",
    "Asn": "N",
    "Asp": "D",
    "Cys": "C",
    "Gln": "Q",
    "Glu": "E",
    "Gly": "G",
    "His": "H",
    "Ile": "I",
    "Leu": "L",
    "Lys": "K",
    "Met": "M",
    "Phe": "F",
    "Pro": "P",
    "Ser": "S",
    "Thr": "T",
    "Trp": "W",
    "Tyr": "Y",
    "Val": "V",
    "Ter": "*",
}


class VcfError(RuntimeError):
    """Raised when a path cannot be read as a VCF at all."""


@dataclass(frozen=True)
class Variant:
    """One alternate allele at one position, named against H37Rv."""

    pos: int
    ref: str
    alt: str
    locus: Locus
    filter: str = "PASS"
    quality: float | None = None
    genotype: str = ""
    depth: int | None = None
    allele_frequency: float | None = None
    # Protein change, e.g. "S450L": worked out against the H37Rv reference, or
    # taken from the VCF's own annotation when it carries one.
    protein_change: str = ""
    annotated_gene: str = ""
    effect: str = ""
    # False when REF disagrees with H37Rv at this position, which means the VCF
    # was called against a different reference or the coordinates are shifted.
    ref_matches: bool = True

    @property
    def gene(self) -> str:
        """Best available gene name: the index's, else the VCF's annotation."""
        return self.locus.gene_name or self.annotated_gene

    @property
    def kind(self) -> str:
        if len(self.ref) == len(self.alt) == 1:
            return "snv"
        return "insertion" if len(self.alt) > len(self.ref) else "deletion"

    @property
    def frameshift(self) -> bool:
        """An indel of non-multiple-of-3 length truncates the protein downstream."""
        return self.kind != "snv" and abs(len(self.alt) - len(self.ref)) % 3 != 0

    @property
    def synonymous(self) -> bool:
        """A silent change: in a resistance gene it is still not resistance."""
        return self.effect == "synonymous"

    def key(self) -> str:
        """Stable identity for a variant: what a cited variant is matched against."""
        return f"{self.pos}{self.ref}>{self.alt}"

    def label(self) -> str:
        """One line naming the variant the way a TB report would.

        e.g. "rpoB S450L (761155 C>T)" or "pncA codon 57 frameshift (2289072 delA)".
        """
        parts = []
        if self.protein_change and self.gene:
            parts.append(f"{self.gene} {self.protein_change}")
        elif self.locus.gene is not None:
            parts.append(self.locus.label())
        else:
            parts.append("intergenic")
        if self.frameshift:
            parts.append("frameshift")
        elif self.kind != "snv":
            parts.append(f"in-frame {self.kind}")
        # Spelled out, because "synonymous" is the difference between evidence
        # of resistance and a silent passenger variant in the same codon.
        elif self.effect in ("synonymous", "nonsense"):
            parts.append(self.effect)
        detail = f"{self.pos} {self.ref}>{self.alt}"
        if self.allele_frequency is not None and self.allele_frequency < 0.9:
            detail += f", AF {self.allele_frequency:.2f}"
        return f"{' '.join(parts)} ({detail})"

    def as_dict(self) -> dict:
        """JSON-safe record, kept beside the prediction so a trace stays checkable."""
        return {
            "key": self.key(),
            "pos": self.pos,
            "ref": self.ref,
            "alt": self.alt,
            "gene": self.gene,
            "region": self.locus.region,
            "codon": self.locus.codon,
            "offset": self.locus.offset,
            "kind": self.kind,
            "frameshift": self.frameshift,
            "synonymous": self.synonymous,
            "ref_matches": self.ref_matches,
            "protein_change": self.protein_change,
            "effect": self.effect,
            "allele_frequency": self.allele_frequency,
            "label": self.label(),
        }


@dataclass
class IsolateVariants:
    """What one VCF yielded: the panel-relevant variants, plus how it was read."""

    isolate: str
    path: str
    variants: list[Variant] = field(default_factory=list)
    total_records: int = 0
    filtered_out: int = 0
    annotated: bool = False
    ref_mismatches: int = 0

    def by_tier(self, panel: Panel) -> list[Variant]:
        """Panel variants, tier 1 first, then by position — the order shown to the LLM."""
        return sorted(
            self.variants,
            key=lambda v: (panel.tier_of(v.gene) or 3, v.pos),
        )


def _open_text(path: Path) -> io.TextIOBase:
    """VCFs arrive plain or bgzipped; both read as text."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", errors="replace")
    return path.open("rt", errors="replace")


def _info_dict(field_text: str) -> dict[str, str]:
    info: dict[str, str] = {}
    for item in field_text.split(";"):
        if not item:
            continue
        key, _, value = item.partition("=")
        info[key] = value
    return info


def _as_float(value: str | None) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _as_int(value: str | None) -> int | None:
    try:
        return int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _short_protein_change(hgvs: str) -> str:
    """ "p.Ser450Leu" -> "S450L"; returns "" when it is not a simple substitution."""
    match = re.fullmatch(
        r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})", (hgvs or "").strip()
    )
    if not match:
        return ""
    before, number, after = match.groups()
    if before not in AMINO_ACIDS or after not in AMINO_ACIDS:
        return ""
    return f"{AMINO_ACIDS[before]}{number}{AMINO_ACIDS[after]}"


def _annotation(info: dict[str, str]) -> tuple[str, str, str]:
    """(gene, protein change, effect) from an snpEff ANN or VEP CSQ field, if any.

    Only the first (most severe) annotation is read; the rest describe the same
    variant against other transcripts, which M. tuberculosis does not have.
    """
    raw = info.get("ANN") or info.get("EFF") or info.get("CSQ")
    if not raw:
        return "", "", ""
    fields = raw.split(",")[0].split("|")
    if len(fields) < 4:
        return "", "", ""
    effect, gene = fields[1].strip(), fields[3].strip()
    hgvs = next(
        (f for f in fields if f.strip().startswith("p.")),
        "",
    )
    return gene, _short_protein_change(hgvs), effect


def _sample_fields(columns: list[str]) -> tuple[str, int | None, float | None]:
    """(genotype, depth, allele frequency) from the FORMAT/sample pair, if present."""
    if len(columns) < 10:
        return "", None, None
    keys, values = columns[8].split(":"), columns[9].split(":")
    sample = dict(zip(keys, values))
    genotype = sample.get("GT", "")
    depth = _as_int(sample.get("DP"))
    frequency = _as_float(sample.get("AF"))
    if frequency is None and sample.get("AD"):
        counts = [_as_int(part) or 0 for part in sample["AD"].split(",")]
        if len(counts) >= 2 and sum(counts):
            frequency = round(sum(counts[1:]) / sum(counts), 4)
    return genotype, depth, frequency


def read_vcf(
    path: str | Path,
    genes: set[str] | None = None,
    pass_only: bool = True,
    min_allele_frequency: float = 0.0,
) -> IsolateVariants:
    """Parse a VCF into named variants, keeping only `genes` when given.

    `genes` holds lowercased gene names and locus tags; None keeps everything.
    """
    path = Path(path)
    if not path.exists():
        raise VcfError(f"VCF not found: {path}")

    index = genome_index()
    result = IsolateVariants(isolate=path.name.split(".")[0], path=str(path))
    try:
        handle = _open_text(path)
    except OSError as e:
        raise VcfError(f"could not open {path}: {e}") from e

    with handle:
        for line in handle:
            if line.startswith("##"):
                if line.startswith(
                    ("##INFO=<ID=ANN", "##INFO=<ID=CSQ", "##INFO=<ID=EFF")
                ):
                    result.annotated = True
                continue
            if line.startswith("#"):
                continue
            columns = line.rstrip("\n").split("\t")
            if len(columns) < 8:
                continue
            result.total_records += 1

            position = _as_int(columns[1])
            if position is None:
                continue
            filter_value = columns[6].strip()
            if pass_only and filter_value.upper() not in PASS_VALUES:
                result.filtered_out += 1
                continue
            info = _info_dict(columns[7])
            genotype, depth, frequency = _sample_fields(columns)
            if genotype in REF_GENOTYPES and genotype:
                result.filtered_out += 1
                continue
            annotated_gene, protein_change, effect = _annotation(info)
            locus = index.locate(position)
            if genes is not None:
                names = {locus.gene_name.lower(), annotated_gene.lower()}
                if locus.gene is not None:
                    names.add(locus.gene.locus_tag.lower())
                if not names & genes:
                    continue

            reference = columns[3].strip()
            for alternate in columns[4].split(","):
                alternate = alternate.strip()
                if not alternate or alternate in (".", "<NON_REF>", "*"):
                    continue
                # Worked out from the reference, so an unannotated VCF still
                # distinguishes a missense change from a silent one.
                called_change, called_effect = consequence(
                    locus, position, reference, alternate
                )
                matches = ref_matches_reference(position, reference)
                if not matches:
                    result.ref_mismatches += 1
                if (
                    frequency is not None
                    and min_allele_frequency
                    and frequency < min_allele_frequency
                ):
                    result.filtered_out += 1
                    continue
                result.variants.append(
                    Variant(
                        pos=position,
                        ref=reference,
                        alt=alternate,
                        locus=locus,
                        filter=filter_value or "PASS",
                        quality=_as_float(columns[5]),
                        genotype=genotype,
                        depth=depth,
                        allele_frequency=frequency,
                        protein_change=protein_change or called_change,
                        annotated_gene=annotated_gene,
                        effect=effect or called_effect,
                        ref_matches=matches,
                    )
                )
    return result


def ref_matches_reference(pos: int, ref: str) -> bool:
    """Whether a record's REF agrees with H37Rv there; True when no reference is loaded."""
    sequence = reference_sequence()
    if not sequence or not ref:
        return True
    return sequence[pos - 1 : pos - 1 + len(ref)].upper() == ref.upper()


def panel_gene_set(panel: Panel) -> set[str]:
    """The lowercased gene names and locus tags a panel accepts, for read_vcf."""
    index = genome_index()
    names: set[str] = set()
    for name in panel.genes:
        names.add(name.lower())
        gene = index.gene(name)
        if gene is not None:
            names.add(gene.locus_tag.lower())
            if gene.gene:
                names.add(gene.gene.lower())
    return names
