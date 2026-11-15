"""Place a genome position on an H37Rv gene, so a bare VCF row can be named."""

import bisect
import gzip
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GENE_TABLE = DATA_DIR / "h37rv_genes.tsv"
REFERENCE = DATA_DIR / "h37rv.fna.gz"

COMPLEMENT = str.maketrans("ACGTacgt", "TGCAtgca")

# Translation table 11 (bacterial). Only the start codons differ from the
# standard table, and a variant is never called against a start codon here.
CODON_TABLE = {
    "TTT": "F",
    "TTC": "F",
    "TTA": "L",
    "TTG": "L",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    "ATG": "M",
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "TAT": "Y",
    "TAC": "Y",
    "TAA": "*",
    "TAG": "*",
    "CAT": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "AAT": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "TGT": "C",
    "TGC": "C",
    "TGA": "*",
    "TGG": "W",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "AGT": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}

# A variant this far upstream of a start codon sits in the promoter, where TB
# resistance really does arise (inhA, ahpC, eis, pncA, Rv0678), so it is
# reported as gene-adjacent rather than dropped as intergenic.
PROMOTER_WINDOW = 200


@dataclass(frozen=True)
class Gene:
    locus_tag: str
    gene: str
    start: int  # 1-based inclusive, forward strand, start <= end
    end: int
    strand: str
    biotype: str = "CDS"

    @property
    def name(self) -> str:
        """The name to show: the gene symbol, falling back to the locus tag."""
        return self.gene or self.locus_tag

    @property
    def coding(self) -> bool:
        """rrs and rrl are drug targets but have no codons; they are numbered by base."""
        return self.biotype == "CDS"

    def codon_of(self, pos: int) -> int | None:
        """1-based codon number for a coding position, counted from the start codon.

        Reverse-strand genes are counted from `end` inwards. None outside the
        gene, and None for RNA genes, which have no reading frame.
        """
        if not self.start <= pos <= self.end or not self.coding:
            return None
        offset = pos - self.start if self.strand == "+" else self.end - pos
        return offset // 3 + 1

    def offset_of(self, pos: int) -> int:
        """Position relative to the start codon: 1-based inside, negative upstream.

        Matches how TB promoter variants are written (e.g. inhA -15, pncA -11).
        """
        if self.strand == "+":
            return pos - self.start + 1 if pos >= self.start else pos - self.start
        return self.end - pos + 1 if pos <= self.end else self.end - pos


@dataclass(frozen=True)
class Locus:
    """Where a position landed: a gene, and how it sits relative to that gene."""

    gene: Gene | None
    region: str  # "coding", "promoter", or "intergenic"
    offset: int | None  # position relative to the start codon
    codon: int | None  # codon number, coding positions only

    @property
    def gene_name(self) -> str:
        return self.gene.name if self.gene else ""

    def label(self) -> str:
        """Short human label, e.g. "rpoB codon 450", "rrs 1401", "fabG1 -15 (promoter)"."""
        if self.gene is None:
            return "intergenic"
        if self.region != "coding":
            return f"{self.gene.name} {self.offset} (promoter)"
        if self.codon is None:  # an RNA gene: number it by base, as rrs 1401 is
            return f"{self.gene.name} {self.offset}"
        return f"{self.gene.name} codon {self.codon}"


def load_genes(path: Path = GENE_TABLE) -> list[Gene]:
    """Read the shipped H37Rv gene table, sorted by start."""
    if not path.exists():
        raise FileNotFoundError(
            f"H37Rv gene table missing at {path}. It ships with the repo under data/."
        )
    genes = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith(("#", "locus_tag\t")):
            continue
        locus, gene, start, end, strand, biotype = line.split("\t")
        genes.append(Gene(locus, gene, int(start), int(end), strand, biotype))
    return sorted(genes, key=lambda g: g.start)


class GenomeIndex:
    """Position -> gene lookup over the H37Rv gene table."""

    def __init__(self, genes: list[Gene] | None = None):
        self.genes = genes if genes is not None else load_genes()
        self._starts = [gene.start for gene in self.genes]
        self._by_name: dict[str, Gene] = {}
        for gene in self.genes:
            self._by_name.setdefault(gene.locus_tag.lower(), gene)
            if gene.gene:
                self._by_name.setdefault(gene.gene.lower(), gene)

    def gene(self, name: str) -> Gene | None:
        """Look a gene up by symbol or locus tag, case-insensitively."""
        return self._by_name.get((name or "").strip().lower())

    def locate(self, pos: int) -> Locus:
        """The gene a position falls in, else the gene it sits just upstream of.

        Genes overlap in places, so a coding hit always wins over a promoter one.
        """
        # Only genes starting at or before pos can contain it; a promoter hit is
        # within PROMOTER_WINDOW of a later start, so look a little past pos too.
        high = bisect.bisect_right(self._starts, pos + PROMOTER_WINDOW)
        low = bisect.bisect_left(self._starts, pos - 30_000)  # longest M. tb gene
        promoter: Gene | None = None
        distance = PROMOTER_WINDOW + 1
        for gene in self.genes[low:high]:
            if gene.start <= pos <= gene.end:
                return Locus(gene, "coding", gene.offset_of(pos), gene.codon_of(pos))
            upstream = gene.start - pos if gene.strand == "+" else pos - gene.end
            # Nearest start codon wins: the fabG1/inhA promoter variants sit a
            # few bases upstream of fabG1 but inside another gene's window too.
            if 0 < upstream < distance:
                promoter, distance = gene, upstream
        if promoter is not None:
            return Locus(promoter, "promoter", promoter.offset_of(pos), None)
        return Locus(None, "intergenic", None, None)


@lru_cache(maxsize=1)
def genome_index() -> GenomeIndex:
    """The shared index; the table is a few thousand rows, so load it once."""
    return GenomeIndex()


@lru_cache(maxsize=1)
def reference_sequence(path: Path = REFERENCE) -> str:
    """The H37Rv chromosome as one uppercase string, 1-based when indexed from 0.

    Returns "" when the reference is not available, which turns protein-level
    naming off rather than failing the run.
    """
    if not path.exists():
        return ""
    with gzip.open(path, "rt") as handle:
        return "".join(
            line.strip().upper() for line in handle if not line.startswith(">")
        ).replace(" ", "")


def translate(codon: str) -> str:
    """One amino-acid letter, or "" when the codon has an ambiguous base."""
    return CODON_TABLE.get(codon.upper(), "")


def consequence(locus: Locus, pos: int, ref: str, alt: str) -> tuple[str, str]:
    """(protein change, effect) for a variant, worked out against the reference.

    e.g. ("S450L", "missense") or ("L452L", "synonymous"). The protein change is
    "" for anything that cannot be named at the residue level — an indel, an
    RNA gene, a promoter variant, or a missing reference.
    """
    gene = locus.gene
    if gene is None:
        return "", ""
    if locus.region == "promoter":
        return "", "upstream"
    if not gene.coding:
        return "", "non_coding"

    if len(ref) != len(alt):
        shift = abs(len(alt) - len(ref)) % 3 != 0
        kind = "insertion" if len(alt) > len(ref) else "deletion"
        return "", ("frameshift" if shift else f"inframe_{kind}")
    if len(ref) != 1:  # a multi-base substitution can span codons
        return "", "substitution"

    sequence = reference_sequence()
    codon_number = locus.codon
    if not sequence or codon_number is None:
        return "", ""

    # The three reference bases of this codon, read 5'->3' on the gene's strand.
    if gene.strand == "+":
        start = gene.start + (codon_number - 1) * 3
        codon = sequence[start - 1 : start + 2]
        offset = pos - start
        new_base = alt.upper()
    else:
        end = gene.end - (codon_number - 1) * 3
        codon = sequence[end - 3 : end].translate(COMPLEMENT)[::-1]
        offset = end - pos
        new_base = alt.upper().translate(COMPLEMENT)
    if len(codon) != 3 or not 0 <= offset <= 2:
        return "", ""

    before = translate(codon)
    mutated = codon[:offset] + new_base + codon[offset + 1 :]
    after = translate(mutated)
    if not before or not after:
        return "", ""
    if before == after:
        return f"{before}{codon_number}{after}", "synonymous"
    if after == "*":
        return f"{before}{codon_number}*", "nonsense"
    return f"{before}{codon_number}{after}", "missense"
