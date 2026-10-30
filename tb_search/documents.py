import logging
import re
from pathlib import Path

TEXT_SUFFIXES = (".txt", ".md", ".markdown", ".html", ".htm", ".csv", ".tsv", ".json")
SUPPORTED_SUFFIXES = (".pdf", *TEXT_SUFFIXES)

# Per document ~ title + abstract; sized for small API budgets.
EXCERPT_CHARS = 3000

logging.getLogger("pypdf").setLevel(logging.ERROR)  # pypdf warns exuberantly


def resolve_input(raw: str | None) -> Path | None:
    """Input dir/file given by the user, else ./papers, else ../papers."""
    if raw:
        candidate = Path(raw).expanduser()
        return candidate if candidate.exists() else None
    for candidate in (Path("papers"), Path.cwd().parent / "papers"):
        if candidate.is_dir():
            return candidate
    return None


def collect_docs(input_path: Path) -> list[Path]:
    """All supported documents under input_path, sorted by filename."""
    if input_path.is_file():
        return [input_path]

    docs = []
    for path in sorted(input_path.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix.lower() in SUPPORTED_SUFFIXES:
            docs.append(path)
    return docs


def extract_text(path: Path, max_chars: int = EXCERPT_CHARS) -> str:
    """First max_chars characters of the document, whitespace-collapsed."""
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        try:
            reader = PdfReader(str(path))
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception as e:  # noqa: BLE001 - any pdf parse error → screen as unreadable
            return f"(could not extract text: {e})"
    else:
        try:
            text = path.read_text(errors="ignore")
            if path.suffix.lower() in (".html", ".htm"):
                text = re.sub(r"<[^>]+>", " ", text)
        except Exception as e:  # noqa: BLE001 - unreadable file → screen as unreadable
            return f"(could not read file: {e})"

    return re.sub(r"\s+", " ", text).strip()[:max_chars]
