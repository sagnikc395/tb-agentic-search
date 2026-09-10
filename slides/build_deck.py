"""Build the project slide deck as a .pptx that Google Slides can open.

    uv run python slides/build_deck.py
    .venv/bin/python slides/build_deck.py

Writes `slides/sae-latent-annotation.pptx`. Upload it to Drive and open it with
Google Slides, or use File > Import slides in an existing presentation.

Plain black text on white. No colour, no fills, no decoration: hierarchy comes
from size and weight only.

Every number in the deck comes from a local run of this repository, collected in
`LOCAL_RUN` below so it can be re-checked against the commands on the
verification slide.
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Inches, Pt

OUT = Path(__file__).resolve().parent / "sae-latent-annotation.pptx"

# ---------------------------------------------------------------------------
# Verified locally on 2026-09-10 with .venv/bin/python. See the "Verified
# locally" slide for the exact commands.
# ---------------------------------------------------------------------------

LOCAL_RUN = {
    "catalog_tools": 2602,
    "exposed_tools": 9,
    "reserved_tools": 6,
    "local_tools": 2,
    "fixture_proteins": 8,
    "fixture_latents": 3,
    "offline_clean": "3/3",
    "live_turns": 3,
    "repo_loc": 1524,
    "notes_lines": 864,
    "matches": {"UniProt": 43, "Ensembl": 78, "InterPro": 18, "Reactome": 39},
}

BLACK = RGBColor(0x00, 0x00, 0x00)
GREY = RGBColor(0x66, 0x66, 0x66)
LINE = RGBColor(0xBB, 0xBB, 0xBB)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

SANS = "Arial"
MONO = "Courier New"

W = Inches(13.333)
H = Inches(7.5)
M = Inches(0.75)
CW = W - 2 * M


def new_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = WHITE
    return slide


def box(slide, x, y, w, h):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    return tf


def line(slide, x, y, w, h=Pt(0.75), color=LINE):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    shape.shadow.inherit = False
    return shape


def p(tf, text, *, font=SANS, size=14, bold=False, color=BLACK, after=6,
      spacing=1.25, indent=0, first=False):
    para = tf.paragraphs[0] if first else tf.add_paragraph()
    para.line_spacing = spacing
    para.space_after = Pt(after)
    para.space_before = Pt(0)
    if indent:
        para.paragraph_format.left_indent = Inches(indent)
    run = para.add_run()
    run.text = text
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    return para


def mixed(tf, parts, *, size=14, after=6, spacing=1.25, indent=0, first=False):
    """A paragraph from (text, kwargs) pairs: bold lead-ins, inline code, etc."""
    para = tf.paragraphs[0] if first else tf.add_paragraph()
    para.line_spacing = spacing
    para.space_after = Pt(after)
    para.space_before = Pt(0)
    if indent:
        para.paragraph_format.left_indent = Inches(indent)
    for text, style in parts:
        run = para.add_run()
        run.text = text
        run.font.name = style.get("font", SANS)
        run.font.size = Pt(style.get("size", size))
        run.font.bold = style.get("bold", False)
        run.font.color.rgb = style.get("color", BLACK)
    return para


def heading(slide, title, sub=None):
    """Slide title, a rule beneath it, and an optional standfirst. Returns next y."""
    tf = box(slide, M, Inches(0.55), CW, Inches(0.6))
    p(tf, title, size=26, bold=True, after=0, spacing=1.05, first=True)
    line(slide, M, Inches(1.22), CW, Pt(1.2), BLACK)
    y = Inches(1.45)
    if sub:
        tf = box(slide, M, y, Inches(11.2), Inches(0.7))
        p(tf, sub, size=13, color=GREY, after=0, spacing=1.3, first=True)
        y += Inches(0.3) + Inches(0.22) * (len(sub) // 115 + 1)
    return y


def page_number(slide, n, total):
    tf = box(slide, W - M - Inches(1.4), H - Inches(0.55), Inches(1.4), Inches(0.25))
    para = p(tf, f"{n} / {total}", font=MONO, size=10, color=GREY, after=0, first=True)
    para.alignment = 2  # right


def bullets(slide, x, y, w, items, *, size=13, after=9, spacing=1.28):
    """Items: plain strings, or (bold lead, rest) pairs."""
    tf = box(slide, x, y, w, H - y - Inches(0.7))
    for i, item in enumerate(items):
        if isinstance(item, tuple):
            mixed(
                tf,
                [(item[0], {"bold": True, "size": size}),
                 (item[1], {"size": size})],
                after=after,
                spacing=spacing,
                first=(i == 0),
            )
        else:
            p(tf, item, size=size, after=after, spacing=spacing, first=(i == 0))
    return tf


def dashed(slide, x, y, w, items, *, size=13, after=8):
    """A list marked with em dashes rather than bullet glyphs."""
    return bullets(slide, x, y, w, [("—  " + i[0], i[1]) if isinstance(i, tuple)
                                    else "—  " + i for i in items], size=size, after=after)


def table(slide, x, y, w, cols, data, *, size=11.5, row_h=0.34, head=True):
    """A hairline table: header rule, thin row rules, no fills."""
    widths = [Inches(c[1]) for c in cols]
    cur = y
    if head:
        cx = x
        for (label, _wi, _st), width in zip(cols, widths):
            tf = box(slide, cx, cur, width - Inches(0.12), Inches(0.22))
            p(tf, label.upper(), font=MONO, size=9.5, color=GREY, after=0, first=True)
            cx += width
        cur += Inches(0.26)
        line(slide, x, cur, w, Pt(1.0), BLACK)
        cur += Inches(0.08)

    for row in data:
        h = Inches(row_h)
        cx = x
        for value, (_label, _wi, style), width in zip(row, cols, widths):
            tf = box(slide, cx, cur + Inches(0.04), width - Inches(0.12), h)
            p(
                tf,
                str(value),
                font=style.get("font", SANS),
                size=style.get("size", size),
                bold=style.get("bold", False),
                color=style.get("color", BLACK),
                after=0,
                spacing=1.15,
                first=True,
            )
            cx += width
        cur += h
        line(slide, x, cur, w, Pt(0.75), LINE)
        cur += Inches(0.06)
    return cur


def mono_block(slide, x, y, w, lines, *, size=11, indent_rule=True):
    """A monospace block (tool trace, terminal output) with a rule on the left."""
    h = Inches(0.22) * len(lines) + Inches(0.2)
    if indent_rule:
        line(slide, x, y, Pt(1.2), h, BLACK)
    tf = box(slide, x + Inches(0.16), y + Inches(0.08), w - Inches(0.16), h)
    for i, text in enumerate(lines):
        bold = text.startswith("$")
        p(tf, text, font=MONO, size=size, bold=bold, after=2, spacing=1.15,
          color=BLACK if bold or not text.startswith("  ") else GREY, first=(i == 0))
    return y + h


# ---------------------------------------------------------------------------
# Slides
# ---------------------------------------------------------------------------


def s_title(slide, n, total):
    tf = box(slide, M, Inches(2.35), Inches(11.4), Inches(1.8))
    p(tf, "Agents for ToolUniverse", size=40, bold=True, after=2, spacing=1.05, first=True)
    p(tf, "for SAE feature description", size=40, bold=True, after=0, spacing=1.05)
    line(slide, M, Inches(4.05), Inches(11.4), Pt(1.2), BLACK)
    tf = box(slide, M, Inches(4.25), Inches(11.4), Inches(0.5))
    p(tf, "Progress till now", size=20, color=GREY, after=0, first=True)
    tf = box(slide, M, Inches(5.55), Inches(11.4), Inches(0.6))
    p(tf, "Sagnik Chatterjee", font=MONO, size=12, after=3, first=True)
    p(tf, "repo: latent-annotation  ·  10 September 2026", font=MONO, size=12, color=GREY, after=0)


def s_done(slide, n, total):
    heading(
        slide,
        "What has been done so far",
        "Four increments of work, from a tool-search script to a complete annotation "
        "pipeline that runs end to end against live databases.",
    )
    table(
        slide,
        M,
        Inches(2.3),
        Inches(11.4),
        [("stage", 2.5, {"size": 12, "bold": True}),
         ("delivered", 8.9, {"size": 12})],
        [
            ("Tool search",
             "ToolUniverse catalog loaded and searchable by pattern; OpenAI tool specs printed "
             "for any tool (main.py)"),
            ("Tool selection",
             f"{LOCAL_RUN['exposed_tools']} protein-context tools chosen from "
             f"{LOCAL_RUN['catalog_tools']:,}, plus {LOCAL_RUN['reserved_tools']} reserved for "
             "mutation-effect work, each mapped to a question the annotation needs answered"),
            ("Agent pipeline",
             "Evidence extraction, tool dispatch, the per-latent tool-use loop, the annotation "
             "schema with client-side validation and repair, and JSONL output with a full "
             "tool-call trace"),
            ("Runnable demo",
             f"Synthetic fixture over {LOCAL_RUN['fixture_proteins']} real accessions and "
             f"{LOCAL_RUN['fixture_latents']} latent patterns, an offline mock client, and a "
             "HuggingFace Qwen-3 client behind the same interface"),
        ],
        row_h=0.72,
        size=12,
    )
    bullets(
        slide,
        M,
        Inches(5.75),
        Inches(11.4),
        [
            ("Confirmed working today. ",
             f"{LOCAL_RUN['catalog_tools']:,} tools load; the offline run produces "
             f"{LOCAL_RUN['offline_clean']} schema-clean records; a live UniProt lookup on "
             "P00533 returns through the toolbox without error."),
            ("Not done yet. ",
             "No real SAE activations, no evaluation of annotation quality, and no test suite. "
             "Nothing here is a scientific result."),
        ],
        size=12.5,
        after=9,
    )
    page_number(slide, n, total)


def s_problem(slide, n, total):
    heading(
        slide,
        "The problem",
        "A protein language model's residue vectors are dense and polysemantic. A sparse "
        "autoencoder splits them into far more, far sparser latents, each firing on a small "
        "subset of residues. Testing whether those latents are real features requires a "
        "description of every one, and an SAE has 10,000 to 100,000 of them.",
    )
    bullets(
        slide,
        M,
        Inches(2.55),
        Inches(11.4),
        [
            ("Showing a model the top sequences and asking for a label does not work. ",
             "Three problems make the obvious approach unsound:"),
            ("1.  The model always finds a pattern. ",
             "Given any set of protein fragments, an LLM writes a fluent explanation. "
             "A label with no falsification condition is not a finding."),
            ("2.  Top-k activations are a biased sample. ",
             "They over-represent whatever the corpus over-represents. “Fires on "
             "kinases” may just track the most common fold in the dataset."),
            ("3.  Protein-level facts do not license residue-level claims. ",
             "“These are all kinases” is guilt by association. “These peaks fall "
             "inside the annotated ATP-binding region at positions X, Y, Z” is a claim "
             "that can be checked against the peak index."),
            ("Each one becomes a hard requirement in the output schema, ",
             "not a suggestion in the prompt: disconfirming_evidence, rejected_hypotheses, "
             "and a source_tool + source_id on every claim."),
        ],
        after=13,
    )
    page_number(slide, n, total)


def s_approach(slide, n, total):
    heading(slide, "The approach: make the agent retrieve its evidence, then cite it")
    bullets(
        slide,
        M,
        Inches(1.7),
        Inches(11.4),
        [
            "An LLM can answer “what does a latent firing on GxGxxG mean?” from memory. "
            "That answer leaves no record anyone can inspect. Given a UniProt tool, it has to "
            "name an accession and use the retrieved entry, and a reviewer can re-run the "
            "same lookup and check the claim.",
            ("ToolUniverse ", "(Harvard MIMS) puts thousands of scientific database endpoints "
             "behind one interface: load the catalog, request an OpenAI-shaped tool spec, call "
             "a tool by name with a dictionary of arguments."),
        ],
        after=11,
    )

    y = Inches(3.35)
    tf = box(slide, M, y, Inches(11.4), Inches(0.3))
    p(tf, "One unit of evidence: a top-activating window, brackets around the peak residues",
      size=11.5, color=GREY, after=0, first=True)
    mono_block(
        slide,
        M,
        Inches(3.68),
        Inches(11.4),
        [
            "L-0142   P00533  peak 4    LG[SGAFG]TVYKGLWIPEGEKVKIPVAIKE",
            "L-0142   P62826  peak 19   MAAQGEPQVQFKLVLVG[DGGTG]KTTFVKRHLTG",
        ],
        size=12,
    )

    table(
        slide,
        M,
        Inches(4.75),
        Inches(11.4),
        [("figure", 2.2, {"font": MONO, "size": 13, "bold": True}),
         ("meaning", 9.2, {"size": 12.5})],
        [
            (f"{LOCAL_RUN['catalog_tools']:,}", "tools in the ToolUniverse catalog, loaded locally"),
            (str(LOCAL_RUN["exposed_tools"]), "database tools exposed to the annotation agent"),
            (str(LOCAL_RUN["local_tools"]), "local tools that serve SAE evidence from this process"),
        ],
        row_h=0.36,
    )
    page_number(slide, n, total)


def s_pipeline(slide, n, total):
    heading(slide, "Pipeline, end to end")
    mono_block(
        slide,
        M,
        Inches(1.7),
        Inches(11.4),
        [
            "protein sequences + activation dump",
            "    -> top activation windows and firing statistics      evidence.py",
            "    -> local evidence tool + curated ToolUniverse tools   tools.py",
            "    -> explicit model/tool loop, one latent per run       agent.py",
            "    -> JSON extraction and schema validation             schema.py",
            "    -> annotations.jsonl                                 one record per latent",
        ],
        size=12.5,
    )

    tf = box(slide, M, Inches(3.5), Inches(11.4), Inches(0.3))
    p(tf, "Three seams, deliberately placed. Swapping the SAE, the tool surface, or the model "
          "touches exactly one of them.", size=13, after=0, first=True)

    table(
        slide,
        M,
        Inches(4.05),
        Inches(11.4),
        [("seam", 3.0, {"font": MONO, "size": 12}),
         ("purpose", 8.4, {"size": 12.5})],
        [
            ("LatentEvidenceStore", "Any SAE export can feed the pipeline. No torch, no numpy: plain Python lists."),
            ("ToolBox", "Local functions and remote database endpoints look identical to the model."),
            ("ChatClient", "The same loop runs on the offline mock or a HuggingFace-served Qwen-3."),
        ],
        row_h=0.36,
    )

    tf = box(slide, M, Inches(5.85), Inches(11.4), Inches(0.5))
    p(tf, "Each output record carries the annotation, the full tool-call trace, the turn count, "
          "and token usage, so any claim can be audited after the fact.",
      size=12, color=GREY, after=0, first=True)
    page_number(slide, n, total)


def s_repo(slide, n, total):
    heading(
        slide,
        "What is in the repository",
        f"{LOCAL_RUN['repo_loc']:,} lines of Python plus {LOCAL_RUN['notes_lines']} lines of "
        "design notes. The project does not train a protein model or an SAE: its input "
        "boundary is a set of sequences plus per-residue activation vectors.",
    )
    table(
        slide,
        M,
        Inches(2.35),
        Inches(11.4),
        [("file", 3.5, {"font": MONO, "size": 11.5}),
         ("lines", 0.85, {"font": MONO, "size": 11.5, "color": GREY}),
         ("role", 7.05, {"size": 11.5})],
        [
            ("latent_annotation/config.py", "98", "Paths, model defaults, loop limits, windowing, tool allowlists"),
            ("latent_annotation/evidence.py", "192", "Activation vectors to top windows, peaks, firing statistics"),
            ("latent_annotation/tools.py", "181", "LocalTool and ToolBox: definitions, dispatch, truncation"),
            ("latent_annotation/llm.py", "345", "ChatClient protocol, HuggingFace Qwen-3 client, offline mock"),
            ("latent_annotation/agent.py", "195", "The per-latent loop: tool calls, nudge, schema repair"),
            ("latent_annotation/schema.py", "203", "Annotation schema, system prompt, JSON recovery, validation"),
            ("main.py", "65", "ToolUniverse discovery: search the catalog, print a tool spec"),
            ("examples/make_fixture.py", "126", "Synthetic activations over 8 real accessions, 3 latent patterns"),
            ("examples/annotate_latents.py", "119", "End-to-end CLI writing one JSONL record per latent"),
            ("notes/01-06", "864", "Agent design, tool selection, SAE research plan, roadmap, deep dive"),
        ],
        row_h=0.315,
        size=11.5,
    )
    page_number(slide, n, total)


def s_evidence(slide, n, total):
    heading(
        slide,
        "Evidence layer: what the agent is allowed to see",
        "Each run covers one latent. The model pulls evidence through a tool call rather than "
        "receiving the whole store in its prompt, so prompt size does not grow with the number "
        "of latents and evidence cannot leak between runs.",
    )
    mono_block(
        slide,
        M,
        Inches(2.5),
        Inches(6.7),
        [
            "L-0142  EGFR  LG[SGAFG]TVYKGLWIPEGEKVKIPVAIKE",
            "L-0142  INSR  LG[QGSFG]MVYEGNARDIIKGEAETRVAVK",
            "L-0731  HBB   VHLT[PEEKS]AVTALWGKVNVDEVGGEALGRL",
        ],
        size=11.5,
    )
    tf = box(slide, M, Inches(3.55), Inches(6.7), Inches(1.0))
    p(tf, "Alongside the windows: firing rate, mean active activation, max activation, "
          "proteins scanned, and the unique accessions that become inputs to the database tools.",
      size=12, color=GREY, after=0, spacing=1.3, first=True)

    table(
        slide,
        M + Inches(7.1),
        Inches(2.5),
        Inches(4.3),
        [("setting", 2.35, {"font": MONO, "size": 11}),
         ("value", 0.75, {"font": MONO, "size": 11}),
         ("", 1.2, {"size": 11, "color": GREY})],
        [
            ("TOP_K_EXAMPLES", "8", "windows"),
            ("WINDOW_SIZE", "24", "residues"),
            ("HIGHLIGHT_SPAN", "2", "residues"),
            ("ACTIVE_THRESHOLD", "> 0.0", "firing"),
            ("TOOL_RESULT_LIMIT", "8,000", "chars"),
            ("max_turns", "12", "per latent"),
        ],
        row_h=0.3,
        size=11,
    )

    bullets(
        slide,
        M,
        Inches(4.75),
        Inches(11.4),
        [
            ("Known limitation. ",
             "Only each protein's single maximum is retained, so eight examples means eight "
             "proteins, not eight independent firing sites. The examples are an extreme, "
             "protein-level sample with no diversity criterion beyond protein identity."),
            ("Not yet checked. ",
             "Activation vectors are not validated against sequence length, residue alphabet, "
             "or finite values, and the peak index is zero-based without saying so in the output."),
        ],
        size=12.5,
        after=10,
    )
    page_number(slide, n, total)


def s_tools(slide, n, total):
    heading(slide, "Tool surface: nine tools, each answering a different question")
    table(
        slide,
        M,
        Inches(1.7),
        Inches(7.4),
        [("tooluniverse tool", 4.4, {"font": MONO, "size": 11}),
         ("answers", 3.0, {"size": 11.5})],
        [
            ("UniProt_get_recommended_name", "what protein is this"),
            ("UniProt_get_function", "what does it do"),
            ("UniProt_get_features", "do the peaks fall in a feature"),
            ("UniProt_get_subcellular_location", "is it tracking localization"),
            ("UniProt_get_ptm_processing", "is it a modification site"),
            ("InterPro_get_entries_for_protein", "which domains"),
            ("InterPro_get_domain_architecture", "domain order and boundaries"),
            ("Pfam_get_protein_annotations", "family assignment"),
            ("Reactome_map_uniprot_to_pathways", "pathway context"),
        ],
        row_h=0.3,
        size=11.5,
    )

    x = M + Inches(7.9)
    w = Inches(4.5)
    bullets(
        slide,
        x,
        Inches(1.7),
        w,
        [
            ("The load-everything trap. ",
             f"The full catalog would add roughly a megabyte to every request, and dozens of "
             f"tools share near-identical descriptions. The local catalog holds "
             f"{LOCAL_RUN['matches']['UniProt']} UniProt tools, "
             f"{LOCAL_RUN['matches']['Ensembl']} matching Ensembl, "
             f"{LOCAL_RUN['matches']['Reactome']} Reactome."),
            ("Residue level, not protein level. ",
             "UniProt_get_features_by_accession is the tool that makes positional claims "
             "checkable: it returns the annotated regions the peak index is compared against."),
            ("Reserved for later. ",
             f"{LOCAL_RUN['reserved_tools']} mutation-effect tools are selected but not wired "
             "to any agent: disease variants, MaveDB, ClinVar, gnomAD constraint."),
            ("Discovery is separate. ",
             "main.py loads the whole catalog and searches it. It is not part of annotation runs."),
        ],
        size=11.5,
        after=10,
    )
    page_number(slide, n, total)


def s_loop(slide, n, total):
    heading(slide, "The agent loop, kept explicit because its failures are the experiment")
    mono_block(
        slide,
        M,
        Inches(1.7),
        Inches(6.6),
        [
            "user:  annotate L-0142",
            "  -> tool_call   get_latent_evidence",
            "  <- windows, accessions, firing stats",
            "  -> tool_call   UniProt_get_features_by_accession",
            "  <- feature table for P00533",
            "  -> tool_call   InterPro_get_entries_for_protein",
            "  <- domain entries",
            "  -> final text  parsed, validated, returned",
        ],
        size=11.5,
    )
    tf = box(slide, M, Inches(3.85), Inches(6.6), Inches(1.4))
    p(tf, "Bookkeeping around seven lines of loop: a fresh transcript per latent, every call "
          "recorded with its arguments and error flag, token usage accumulated, and an "
          "ungrounded answer left visible in the result rather than hidden by a retry.",
      size=12, color=GREY, after=0, spacing=1.3, first=True)

    table(
        slide,
        M + Inches(7.0),
        Inches(1.7),
        Inches(5.4),
        [("open-weight behaviour", 2.9, {"size": 11}),
         ("handling", 2.5, {"size": 11, "color": GREY})],
        [
            ("Answers before reading activations", "One nudge, then recorded"),
            ("Wraps JSON in a fence, adds prose", "extract_json tries 3 candidates"),
            ("Ignores response_format", "Schema validated client-side"),
            ("Tool arguments arrive as a string", "_coerce_arguments normalizes"),
            ("Returns malformed or invalid JSON", "Up to 2 repair turns"),
            ("A tool raises or endpoint 500s", "Visible error result, not a crash"),
        ],
        row_h=0.42,
        size=11,
    )
    page_number(slide, n, total)


def s_schema(slide, n, total):
    heading(
        slide,
        "Output contract",
        "A free-text label cannot be scored, compared across SAE runs, or audited. Validation "
        "runs inside the loop, so downstream code gets a valid object or an explicit error.",
    )
    table(
        slide,
        M,
        Inches(2.25),
        Inches(7.2),
        [("field", 2.6, {"font": MONO, "size": 11}),
         ("requirement", 4.6, {"size": 11.5})],
        [
            ("label", "Eight words or fewer, plain language"),
            ("concept_type", "One of 12 enums, incl. polysemantic, uninterpretable"),
            ("confidence", "high / medium / low"),
            ("evidence[]", "Every claim carries source_tool + source_id"),
            ("rejected_hypotheses[]", "Competing explanations and why each failed"),
            ("disconfirming_evidence", "An observation that would falsify the label"),
            ("mutation_effect", "Relevance, predicted sensitivity, rationale"),
        ],
        row_h=0.34,
        size=11.5,
    )
    bullets(
        slide,
        M + Inches(7.7),
        Inches(2.25),
        Inches(4.7),
        [
            ("Why the enums matter. ",
             "A pipeline that never returns uninterpretable has a calibration problem, not a "
             "discovery. The fixture includes a broad aromatic-residue latent specifically to "
             "test whether the agent will decline to label it."),
            ("Why sensitivity is predicted here. ",
             "mutation_effect.predicted_sensitivity is recorded before the agent has any access "
             "to DMS data. That ordering is what makes it a prediction rather than a retrieval."),
        ],
        size=11.5,
        after=12,
    )
    page_number(slide, n, total)


def s_local(slide, n, total):
    heading(
        slide,
        "Verified locally",
        "Every command below was run against the checked-in fixture on 10 September 2026 using "
        "the project virtualenv.",
    )
    mono_block(
        slide,
        M,
        Inches(2.2),
        Inches(7.0),
        [
            "$ python main.py",
            f"  {LOCAL_RUN['catalog_tools']} tools loaded, 9 exposed, 6 reserved",
            "",
            "$ python examples/annotate_latents.py --no-tooluniverse",
            f"  wrote annotations.jsonl ({LOCAL_RUN['offline_clean']} clean)",
            "",
            "$ python examples/annotate_latents.py L-0142",
            "  tools used : get_latent_evidence,",
            "               UniProt_get_function_by_accession",
            f"  live call on P00533: no error, {LOCAL_RUN['live_turns']} turns, schema clean",
        ],
        size=11,
    )

    table(
        slide,
        M + Inches(7.5),
        Inches(2.2),
        Inches(4.9),
        [("latent", 1.15, {"font": MONO, "size": 11}),
         ("firing", 0.9, {"font": MONO, "size": 11}),
         ("mean", 0.75, {"font": MONO, "size": 11}),
         ("windows", 0.9, {"font": MONO, "size": 11}),
         ("pattern", 1.2, {"size": 11, "color": GREY})],
        [
            ("L-0142", "0.0449", "2.03", "4 / 8", "glycine-rich"),
            ("L-0731", "0.0336", "2.28", "6 / 8", "acidic pairs"),
            ("L-1503", "0.0710", "2.40", "8 / 8", "any aromatic"),
        ],
        row_h=0.32,
        size=11,
    )

    bullets(
        slide,
        M,
        Inches(4.95),
        Inches(11.4),
        [
            ("The broad latent is already visible in the numbers. ",
             "L-1503 fires in all eight proteins with mean activation equal to its maximum, "
             "which is the signature a latent should have when there is no single explanation "
             "to find."),
            ("Read the mock carefully. ",
             "It exercises evidence, tool dispatch, schema validation, and JSONL output, but "
             "performs no biological reasoning. The clean 3/3 is a plumbing result, not a "
             "scientific one."),
        ],
        size=12.5,
        after=10,
    )
    page_number(slide, n, total)


def s_status(slide, n, total):
    heading(slide, "Status: built, and not yet built")
    table(
        slide,
        M,
        Inches(1.7),
        Inches(11.4),
        [("component", 8.0, {"size": 12}),
         ("state", 3.4, {"font": MONO, "size": 11, "color": GREY})],
        [
            ("Evidence extraction from an activation dump", "working"),
            ("ToolUniverse dispatch across 9 curated tools", "working"),
            ("Agent loop: nudge, schema repair, full tool trace", "working"),
            ("Offline mock end to end, 3/3 clean records", "working"),
            ("Live UniProt call through the toolbox", "working"),
            ("Qwen-3 path via HuggingFace Inference API", "implemented, unevaluated"),
            ("Real SAE export instead of the synthetic fixture", "not started"),
            ("Annotation quality evaluation", "not started"),
            ("Test suite", "not started"),
            ("Mutation-effect agent", "tools selected only"),
        ],
        row_h=0.315,
        size=12,
    )
    tf = box(slide, M, Inches(6.1), Inches(11.4), Inches(0.7))
    mixed(
        tf,
        [("The honest summary. ", {"bold": True, "size": 13}),
         ("The scaffold is complete and exercised. What it has not yet done is annotate a real "
          "SAE latent with a real model and have anyone check the answer.", {"size": 13})],
        after=0,
        spacing=1.3,
        first=True,
    )
    page_number(slide, n, total)


def s_mutation(slide, n, total):
    heading(
        slide,
        "Where this is going: mutation-effect prediction",
        "Zero-shot variant effect from a PLM is usually a masked-marginal score. It says how bad "
        "a substitution is, but not which internal features changed. Annotated latents could "
        "supply that decomposition.",
    )
    bullets(
        slide,
        M,
        Inches(2.5),
        Inches(11.4),
        [
            ("A.  Explain an existing score. ",
             "Compute the variant score as usual, then attribute its change across the latents "
             "that differ between wild type and mutant. Does not change the predictor."),
            ("B.  Latents as predictor inputs. ",
             "Fit a small model on latent activations at the mutated position and compare "
             "against the masked-marginal baseline on ProteinGym."),
            ("C.  Annotation-guided priors, the direct test. ",
             "If a latent is labelled active_or_binding_site with high confidence, substitutions "
             "that suppress it should have larger effects. Correlate predicted_sensitivity "
             "against measured DMS effects at the residues each latent fires on."),
            ("A second agent, kept apart. ",
             "MUTATION_EFFECT_TOOLS is already selected but no code assembles that agent yet. "
             "That is deliberate: if the annotation agent can retrieve DMS measurements, the "
             "test measures retrieval, not prediction."),
        ],
        size=12.5,
        after=11,
    )
    tf = box(slide, M, Inches(5.75), Inches(11.4), Inches(1.0))
    p(tf, "Limits to state up front: SAE latents are not stable across training runs, so any "
          "“the catalytic-site latent” claim needs a cross-run check; wider SAEs split one "
          "concept into several; and a better L0 / reconstruction tradeoff does not imply more "
          "interpretable latents.",
      size=12, color=GREY, after=0, spacing=1.3, first=True)
    page_number(slide, n, total)


def s_gaps(slide, n, total):
    heading(slide, "Gaps: what would break if this scaled tomorrow")
    left = [
        ("No evaluation loop. ",
         "Nothing scores the annotations. Held-out activation prediction, random-label controls, "
         "repeat-annotation agreement, and a human-audited subset all have to exist before these "
         "labels are treated as measurements."),
        ("No tests. ",
         "First targets are the evidence and schema seams: highlighting, empty and all-negative "
         "activations, ties, malformed vector lengths, JSON extraction, repair, tool-error routing."),
        ("Grounding is not enforced end to end. ",
         "The loop requires the local evidence call, but the schema does not actually require a "
         "database citation, so a local-only run can validate on activation evidence alone."),
    ]
    right = [
        ("The fixture invites over-reading. ",
         "Its accessions are real; its sequences are illustrative fragments. Positions cannot be "
         "checked against database records without an explicit fragment offset."),
        ("Operationally single-run. ",
         "Sequential, output overwritten, no resume, no persisted ToolUniverse cache, no "
         "throttling. Unsafe for thousands of latents against live EBI and UniProt endpoints."),
        ("Thin reproducibility metadata. ",
         "Records lack fixture hash, model and provider, prompt and schema version, tool catalog "
         "version, and wall time. HuggingFace also routes to a different provider per run."),
    ]
    bullets(slide, M, Inches(1.75), Inches(5.5), left, size=12, after=13)
    bullets(slide, M + Inches(5.9), Inches(1.75), Inches(5.5), right, size=12, after=13)
    page_number(slide, n, total)


def s_next(slide, n, total):
    heading(slide, "Next, in dependency order")
    table(
        slide,
        M,
        Inches(1.7),
        Inches(11.4),
        [("step", 0.6, {"font": MONO, "size": 12}),
         ("work", 6.6, {"size": 12}),
         ("unblocks", 4.2, {"size": 12, "color": GREY})],
        [
            ("1", "Tests and input validation at the evidence and schema seams", "Trusting any later number"),
            ("2", "Fix provider fallback; make offline vs. live explicit in the CLI", "Honest run records"),
            ("3", "Resumable JSONL, cache persistence, throttling, bounded concurrency", "Runs beyond a handful"),
            ("4", "Swap the fixture for a versioned real SAE export with offsets", "Actual biology"),
            ("5", "Evaluation: held-out activation prediction, controls, human audit", "Claiming the labels mean something"),
            ("6", "Build the mutation-effect agent on the reserved tools", "Experiment C"),
        ],
        row_h=0.38,
        size=12,
    )
    tf = box(slide, M, Inches(4.85), Inches(11.4), Inches(1.6))
    p(tf, "Three questions to answer before scaling up", size=13, bold=True, after=8, first=True)
    for text in [
        "Does annotation quality depend more on the tool surface or on the model? Ablate by running with --no-tooluniverse.",
        "How stable are annotations across two SAEs trained on the same activations? If unstable, the unit of analysis has to be coarser.",
        "What fraction of latents come back uninterpretable? If the pipeline never uses that label, its confidence calibration is suspect.",
    ]:
        p(tf, "—  " + text, size=12, after=6, spacing=1.28)
    page_number(slide, n, total)


SLIDES = [
    s_title,
    s_done,
    s_problem,
    s_approach,
    s_pipeline,
    s_repo,
    s_evidence,
    s_tools,
    s_loop,
    s_schema,
    s_local,
    s_status,
    s_mutation,
    s_gaps,
    s_next,
]


def build() -> Path:
    prs = Presentation()
    prs.slide_width = W
    prs.slide_height = H
    total = len(SLIDES)
    for i, builder in enumerate(SLIDES, start=1):
        builder(new_slide(prs), i, total)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    return OUT


if __name__ == "__main__":
    path = build()
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB, {len(SLIDES)} slides)")
