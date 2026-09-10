# Slides

`sae-latent-annotation.pptx` — 15 slides, plain black and white, on the agents /
ToolUniverse work for SAE feature description.

## Open in Google Slides

Upload the `.pptx` to Drive and open it with Google Slides, or use
**File → Import slides** inside an existing presentation. Fonts are Arial and
Courier New, so nothing needs to be substituted on import.

## Regenerate

```bash
uv run python slides/build_deck.py       # or: .venv/bin/python slides/build_deck.py
```

The deck is generated, not hand-edited: change `build_deck.py` and rebuild rather
than editing the `.pptx`, otherwise the next build overwrites the edit.

## Where the numbers come from

Every figure in the deck was taken from a local run against the checked-in
fixture and is collected in `LOCAL_RUN` at the top of `build_deck.py`. To
re-check them:

```bash
.venv/bin/python main.py                                      # catalog size, allowlists
.venv/bin/python examples/annotate_latents.py --no-tooluniverse   # offline, 3/3 clean
.venv/bin/python examples/annotate_latents.py L-0142           # live UniProt lookup
```
