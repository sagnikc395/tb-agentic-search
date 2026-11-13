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


def read_text(path: Path) -> str:
    """The whole document as whitespace-collapsed text."""
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

    return re.sub(r"\s+", " ", text).strip()


def extract_text(path: Path, max_chars: int = EXCERPT_CHARS) -> str:
    """First max_chars characters of the document, whitespace-collapsed."""
    return read_text(path)[:max_chars]


# Section headings as they survive whitespace-collapsing: a capitalised heading
# word followed by the capital that starts the section's first sentence.
METHODS_HEADING = re.compile(
    r"\b(?:Materials and Methods|Methods|Methodology|METHODS)\s+(?=[A-Z])"
)
CLOSING_HEADING = re.compile(
    r"\b(?:Conclusions?|Discussion|CONCLUSIONS?|DISCUSSION)\s+(?=[A-Z])"
)


def section_excerpt(text: str, max_chars: int = EXCERPT_CHARS) -> str:
    """Opening (title + abstract) plus the starts of the methods and closing sections.

    The opening alone can't tell a TB methods paper from one that names TB in
    passing. A section that can't be found gives its share back to the opening.
    """
    if len(text) <= max_chars:
        return text

    share = max_chars // 4
    head_end = max_chars // 2
    parts = []
    methods = METHODS_HEADING.search(text, head_end)
    if methods:
        parts.append(text[methods.start() : methods.start() + share])
    closings = list(
        CLOSING_HEADING.finditer(text, methods.end() if methods else head_end)
    )
    if closings:  # the last: the real section, not an in-text mention
        start = closings[-1].start()
        parts.append(text[start : start + share])

    head = text[: max_chars - share * len(parts)]
    return " [...] ".join([head, *parts])
