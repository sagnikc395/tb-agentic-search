"""Which genes a drug's resistance is read from, and one spelling per drug.

The gene panels follow the WHO catalogue of M. tuberculosis mutations (2nd
edition, 2023): tier 1 genes carry the established resistance mechanism, tier 2
genes are the secondary ones also graded for that drug. Panels bound what the
agent is shown, and grading its cited genes is what makes a reasoning trace
checkable rather than merely fluent.
"""

from dataclasses import dataclass

# The phenotype columns in master_resistance_table.csv, mapped to the drug key
# used everywhere else. Aliases cover the spelling drift in that header row.
DRUG_ALIASES = {
    "AMIKACIN": "amikacin",
    "AMOXICILLIN": "amoxicillin",
    "AMOXICILLIN_CLAVULANATE": "amoxicillin_clavulanate",
    "CAPREOMYCIN": "capreomycin",
    "CIPROFLOXACIN": "ciprofloxacin",
    "CLARITHROMYCIN": "clarithromycin",
    "CLOFAZIMINE": "clofazimine",
    "CYCLOSERINE": "cycloserine",
    "ETHAMBUTOL": "ethambutol",
    "ETHIONAMIDE": "ethionamide",
    "ETHIOMIDE": "ethionamide",
    "GATIFLOXACIN": "gatifloxacin",
    "IMIPENEM": "imipenem",
    "ISONIAZID": "isoniazid",
    "KANAMYCIN": "kanamycin",
    "LEVOFLOXACIN": "levofloxacin",
    "LINEZOLID": "linezolid",
    "MOXIFLOXACIN": "moxifloxacin",
    "NICOTINAMIDE": "nicotinamide",
    "OFLOXACIN": "ofloxacin",
    "OXIFLOXACIN": "ofloxacin",
    "PARA-AMINOSALICYLIC_ACID": "para_aminosalicylic_acid",
    "PARA_AMINOSALICYLIC_ACID": "para_aminosalicylic_acid",
    "PROTHIONAMIDE": "prothionamide",
    "PYRAZINAMIDE": "pyrazinamide",
    "RIFABUTIN": "rifabutin",
    "RIFAMPICIN": "rifampicin",
    "STREPTOMYCIN": "streptomycin",
}

# drug key -> (tier 1 genes, tier 2 genes). Gene names and locus tags both
# resolve through the H37Rv table, so either spelling is fine here.
PANELS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "rifampicin": (("rpoB",), ("rpoA", "rpoC")),
    "rifabutin": (("rpoB",), ("rpoA", "rpoC")),
    "isoniazid": (
        ("katG", "fabG1", "inhA", "ahpC"),
        ("mshA", "ndh", "Rv1258c", "Rv2752c"),
    ),
    "ethambutol": (
        ("embB",),
        ("embA", "embC", "embR", "aftB", "Rv3806c", "Rv3792"),
    ),
    "pyrazinamide": (("pncA",), ("clpC1", "panD", "rpsA")),
    "streptomycin": (("rpsL", "rrs", "gid"), ("whiB7",)),
    "amikacin": (("rrs",), ("eis", "whiB7")),
    "kanamycin": (("rrs", "eis"), ("whiB7",)),
    "capreomycin": (("rrs", "tlyA"), ("whiB7",)),
    "levofloxacin": (("gyrA", "gyrB"), ()),
    "moxifloxacin": (("gyrA", "gyrB"), ()),
    "ofloxacin": (("gyrA", "gyrB"), ()),
    "ciprofloxacin": (("gyrA", "gyrB"), ()),
    "gatifloxacin": (("gyrA", "gyrB"), ()),
    "ethionamide": (
        ("ethA", "inhA", "fabG1"),
        ("ethR", "mshA", "ndh", "Rv0565c"),
    ),
    "prothionamide": (
        ("ethA", "inhA", "fabG1"),
        ("ethR", "mshA", "ndh", "Rv0565c"),
    ),
    "para_aminosalicylic_acid": (
        ("folC", "thyA", "ribD"),
        ("thyX", "Rv2763c"),
    ),
    "cycloserine": (("alr", "ald", "ddl", "cycA"), ()),
    "clofazimine": (("Rv0678",), ("pepQ", "Rv1979c", "mmpL5", "mmpS5")),
    "clarithromycin": (("rrl",), ("Rv1988", "whiB7")),
    "linezolid": (("rplC", "rrl"), ()),
}

# Drugs in the phenotype table that the WHO catalogue does not grade genes for.
# They still get screened, against the union panel, with the prompt saying so.
UNGRADED = ("amoxicillin", "amoxicillin_clavulanate", "imipenem", "nicotinamide")


@dataclass(frozen=True)
class Panel:
    """The genes one prediction is allowed to reason over."""

    drug: str
    tier1: tuple[str, ...]
    tier2: tuple[str, ...]
    # True when no drug-specific panel exists and the union set was substituted.
    is_fallback: bool = False

    @property
    def genes(self) -> tuple[str, ...]:
        return self.tier1 + self.tier2

    def tier_of(self, gene: str) -> int | None:
        """1 or 2 for a panel gene, None for anything else."""
        folded = (gene or "").strip().lower()
        if folded in {name.lower() for name in self.tier1}:
            return 1
        if folded in {name.lower() for name in self.tier2}:
            return 2
        return None


def all_resistance_genes() -> tuple[str, ...]:
    """Every gene named by any panel, deduplicated and ordered."""
    seen: dict[str, None] = {}
    for tier1, tier2 in PANELS.values():
        for gene in tier1 + tier2:
            seen.setdefault(gene, None)
    return tuple(seen)


def drug_key(name: str) -> str:
    """One spelling for a drug, whether it came from a CSV header or a CLI flag."""
    raw = (name or "").strip()
    if not raw:
        return ""
    upper = raw.upper().replace(" ", "_")
    if upper in DRUG_ALIASES:
        return DRUG_ALIASES[upper]
    return raw.lower().replace("-", "_").replace(" ", "_")


def panel_for(drug: str) -> Panel:
    """The gene panel for a drug; the union panel when the drug has none graded."""
    key = drug_key(drug)
    if key in PANELS:
        tier1, tier2 = PANELS[key]
        return Panel(key, tier1, tier2)
    return Panel(key, all_resistance_genes(), (), is_fallback=True)


def known_drugs() -> tuple[str, ...]:
    """Drug keys with a graded panel, alphabetically."""
    return tuple(sorted(PANELS))
