"""Turn data/cleaned_vcf_paths into one labelled dataset per drug.

Two files go in. master_resistance_table.csv carries the phenotypic DST call
(R/S) for each isolate and drug; big_tb_file_paths.txt lists the BIG-TB VCFs on
the cluster. They join on the table's New_ID column, which matches the VCF
filename stem for all 17,942 BIG-TB isolates. The table's own `path` column
points at a different archive (the Farhat lab rollingDB) and is kept only for
reference.
"""

import csv
import random
from dataclasses import dataclass, field
from pathlib import Path

from tb_search.catalogue import drug_key

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "cleaned_vcf_paths"
RESISTANCE_TABLE = "master_resistance_table.csv"
VCF_PATHS = "big_tb_file_paths.txt"

VCF_SUFFIX = "_variants.vcf"
# The column that ends the run of phenotype columns in the table's header.
PHENOTYPE_END = "category"
RESISTANT, SUSCEPTIBLE = "R", "S"


class CohortError(RuntimeError):
    """Raised when the cohort files are missing or cannot be joined."""


@dataclass(frozen=True)
class Isolate:
    """One isolate: its VCF on the cluster, plus every phenotype recorded for it."""

    new_id: str
    name: str
    vcf_path: str
    phenotypes: dict[str, str]  # drug key -> "R" / "S"
    category: str = ""
    biosample: str = ""
    archive_path: str = ""  # the table's own `path` column, a different archive

    def phenotype(self, drug: str) -> str | None:
        return self.phenotypes.get(drug_key(drug))


@dataclass(frozen=True)
class Sample:
    """One row of a per-drug dataset: an isolate, a drug, and the DST label."""

    isolate: str
    drug: str
    phenotype: str  # "R" or "S"
    vcf_path: str

    @property
    def resistant(self) -> bool:
        return self.phenotype == RESISTANT


@dataclass
class Cohort:
    isolates: list[Isolate] = field(default_factory=list)
    drugs: list[str] = field(default_factory=list)  # drug keys, table order
    # Isolates in the table with no BIG-TB VCF, and VCFs with no table row.
    unmatched_rows: int = 0
    unmatched_vcfs: int = 0

    def counts(self, drug: str) -> dict[str, int]:
        """How many R and S isolates a drug has, over isolates that have a VCF."""
        key = drug_key(drug)
        tally = {RESISTANT: 0, SUSCEPTIBLE: 0}
        for isolate in self.isolates:
            call = isolate.phenotypes.get(key)
            if call in tally:
                tally[call] += 1
        return tally

    def summary(self) -> list[dict]:
        """Per-drug dataset sizes, largest first — what `--cohort` prints."""
        rows = []
        for drug in self.drugs:
            tally = self.counts(drug)
            total = tally[RESISTANT] + tally[SUSCEPTIBLE]
            if total:
                rows.append(
                    {
                        "drug": drug,
                        "resistant": tally[RESISTANT],
                        "susceptible": tally[SUSCEPTIBLE],
                        "total": total,
                    }
                )
        return sorted(rows, key=lambda row: -row["total"])

    def dataset(
        self,
        drug: str,
        limit: int | None = None,
        balance: bool = False,
        seed: int = 0,
        vcf_root: str | None = None,
    ) -> list[Sample]:
        """The labelled samples for one drug.

        `balance` takes equal numbers of R and S, which keeps accuracy readable
        on the drugs where susceptible isolates outnumber resistant ones 4:1.
        `vcf_root` rewrites the cluster directory to wherever the VCFs are
        mounted locally.
        """
        key = drug_key(drug)
        groups: dict[str, list[Sample]] = {RESISTANT: [], SUSCEPTIBLE: []}
        for isolate in self.isolates:
            call = isolate.phenotypes.get(key)
            if call not in groups:
                continue
            groups[call].append(
                Sample(
                    isolate=isolate.new_id,
                    drug=key,
                    phenotype=call,
                    vcf_path=remap_path(isolate.vcf_path, vcf_root),
                )
            )

        shuffler = random.Random(seed)
        for samples in groups.values():
            shuffler.shuffle(samples)

        if balance:
            take = min(len(groups[RESISTANT]), len(groups[SUSCEPTIBLE]))
            if limit is not None:
                take = min(take, max(1, limit // 2))
            picked = groups[RESISTANT][:take] + groups[SUSCEPTIBLE][:take]
        else:
            picked = groups[RESISTANT] + groups[SUSCEPTIBLE]
            shuffler.shuffle(picked)
            if limit is not None:
                picked = picked[:limit]
        shuffler.shuffle(picked)
        return picked


def remap_path(path: str, vcf_root: str | None) -> str:
    """Rewrite a cluster VCF path to sit under `vcf_root`, keeping the filename."""
    if not vcf_root:
        return path
    return str(Path(vcf_root) / Path(path).name)


def _vcf_index(path: Path) -> dict[str, str]:
    """{filename stem: full path} for the listed BIG-TB VCFs."""
    index: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        name = Path(line).name
        stem = (
            name[: -len(VCF_SUFFIX)] if name.endswith(VCF_SUFFIX) else Path(line).stem
        )
        index[stem] = line
    return index


def load_cohort(data_dir: Path | str = DATA_DIR) -> Cohort:
    """Join the resistance table to the VCF path list on New_ID."""
    data_dir = Path(data_dir)
    table, paths = data_dir / RESISTANCE_TABLE, data_dir / VCF_PATHS
    for required in (table, paths):
        if not required.exists():
            raise CohortError(
                f"{required.name} not found under {data_dir}. "
                "Both cohort files live in data/cleaned_vcf_paths/."
            )

    vcfs = _vcf_index(paths)
    cohort = Cohort()
    seen: set[str] = set()

    with table.open(newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        if "New_ID" not in columns or PHENOTYPE_END not in columns:
            raise CohortError(
                f"{table.name} is missing the New_ID or {PHENOTYPE_END} column."
            )
        # Phenotypes are the columns between the isolate id and `category`.
        phenotype_columns = columns[1 : columns.index(PHENOTYPE_END)]
        cohort.drugs = [drug_key(name) for name in phenotype_columns]

        for row in reader:
            new_id = (row.get("New_ID") or "").strip()
            vcf_path = vcfs.get(new_id)
            if not vcf_path:
                cohort.unmatched_rows += 1
                continue
            seen.add(new_id)
            phenotypes = {}
            for column in phenotype_columns:
                call = (row.get(column) or "").strip().upper()
                if call in (RESISTANT, SUSCEPTIBLE):
                    phenotypes[drug_key(column)] = call
            cohort.isolates.append(
                Isolate(
                    new_id=new_id,
                    name=(row.get("Isolate") or new_id).strip(),
                    vcf_path=vcf_path,
                    phenotypes=phenotypes,
                    category=(row.get("category") or "").strip(),
                    biosample=(row.get("biosample") or "").strip(),
                    archive_path=(row.get("path") or "").strip(),
                )
            )

    cohort.unmatched_vcfs = len(set(vcfs) - seen)
    if not cohort.isolates:
        raise CohortError(
            "No isolate in the resistance table matched a listed BIG-TB VCF; "
            "check that New_ID still lines up with the VCF filenames."
        )
    return cohort
