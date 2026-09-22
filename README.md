# latent-annotation

Annotate sparse-autoencoder (SAE) latents from protein language models.

For each latent, the agent fetches the sequence windows with the highest
activations. It then checks the proteins in UniProt, InterPro, Pfam, and Reactome
through [ToolUniverse](https://github.com/mims-harvard/ToolUniverse). The result is
a structured annotation with a source for each claim, rejected alternatives, and
an observation that would falsify the label.

## Run it

This repository currently contains a **minimal viable, offline prototype** of the
Stage 0 loop in `PLAN.md`. It is intentionally not installable and does not claim
biological validity yet: structure, ESM scores, SAE activations, and hypotheses are
mocked so the orchestration, deterministic annotation, citations, and trace can be
shown to an advisor before GPU/API work.

Run it directly from the repository root:

```bash
python examples/annotate_latents.py --dry-run --mut L7D
python examples/annotate_latents.py --dry-run --mut L7D --json
```

To use the supplied DNAJA1/InterPro data path through ToolUniverse:

```bash
python examples/annotate_latents.py --tooluniverse --protein DNAJA1 --mut L7D
```

This calls `UniProt_search`, `UniProt_get_sequence_by_accession`,
`UniProt_get_features_by_accession`, `InterPro_get_protein_domains`, and
`InterPro_get_residue_annotations` for reviewed human DNAJA1 (`P31689`). It fails loudly on missing ToolUniverse or network/API errors, and records (rather
than hides) numbering mismatches. For example, the live UniProt sequence reports
Y7 while the plan’s construct mutation is L7D; that discrepancy is now visible in
the report and should be resolved before interpreting the mutation biologically.
The current slice still leaves ESM, SAE activations, and 3D contact computation for
the next stage; it does not present those as real evidence.

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
