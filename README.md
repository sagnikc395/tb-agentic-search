# latent-annotation

Annotate sparse-autoencoder (SAE) latents from protein language models.

For each latent, the agent fetches the sequence windows with the highest
activations. It then checks the proteins in UniProt, InterPro, Pfam, and Reactome
through [ToolUniverse](https://github.com/mims-harvard/ToolUniverse). The result is
a structured annotation with a source for each claim, rejected alternatives, and
an observation that would falsify the label.

## Run it

```bash
uv sync
uv run python examples/annotate_latents.py                        # mock model, offline
uv run python examples/annotate_latents.py --show-evidence        # no model at all
```

For real annotations, copy `.env.example` to `.env`, set `HF_TOKEN`, and:

```bash
uv run python examples/annotate_latents.py --provider huggingface
uv run python examples/annotate_latents.py --provider huggingface --model Qwen/Qwen3-32B
```

The default model client is `MockChatClient`. It follows the same path as a model:
fetch evidence, look up the first accession, and return an annotation that matches
the schema. It needs no model token or model API call. ToolUniverse requests still
run, so the mock covers the full pipeline except biological interpretation.

## Layout

```
latent_annotation/
  config.py     tool lists, model defaults, evidence windowing, and loop limits
  evidence.py   SAE activations -> top-activating windows, firing stats (no torch/numpy)
  tools.py      ToolUniverse + local tools behind one dispatcher
  llm.py        ChatClient protocol, Qwen-3 through HuggingFace, and the mock
  agent.py      the tool-use loop
  schema.py     annotation JSON schema, system prompt, JSON extraction/validation
examples/
  make_fixture.py       synthetic activations with real accessions
  annotate_latents.py   end-to-end demo
notes/          design notes, research directions, roadmap, and codebase map
```

`latent_annotation/__init__.py` is empty. Import from the defining submodule, for
example `from latent_annotation.evidence import LatentEvidenceStore`.

## Using your own SAE

The only coupling point is `LatentEvidenceStore`:

```python
from latent_annotation.evidence import LatentEvidenceStore, ProteinRecord

store = LatentEvidenceStore.from_activation_dump(
    proteins=[ProteinRecord("EGFR", sequence, accession="P00533")],
    activations={"L-0142": {"EGFR": [0.0, 0.0, 2.4, ...]}},  # per-residue, per latent
)
```

The rest of the pipeline does not depend on how the activations were produced.

## Notes

See [`notes/`](notes/README.md) for the design decisions and research plan.
