# Using ToolUniverse

## What it is

ToolUniverse puts about 2,600 scientific database endpoints behind one interface.
Load the catalog, request an OpenAI-shaped specification, then call a tool by name
with a dictionary of arguments. `main.py` covers discovery and specification
lookup. The annotation agent uses only a small subset of the catalog.

```python
tu = ToolUniverse(); tu.load_tools()
tu.find_tools_by_pattern("UniProt")          # discovery
tu.tool_specification(name, format="openai") # spec -> straight into the tools array
tu.run({"name": name, "arguments": {...}})   # execution
```

`ToolBox._as_function` wraps each specification as
`{"type": "function", "function": ...}` for Qwen-3.

## The load-everything trap

`load_tools()` gives you the full catalog. Do not pass the catalog to the model:

- The schemas would add roughly a megabyte to every request.
- Dozens of tools have similar descriptions, so the model has trouble choosing
  (there are 43 UniProt tools and 78 matching "Ensembl").
- Latency and cost increase on every turn.

Load the catalog during development, but expose only the selected tools at runtime.
`PROTEIN_CONTEXT_TOOLS` contains nine tools. Each answers a different question
about the proteins behind a latent's strongest activations. A task that needs other
tools should get a separate configuration instead of expanding every annotation
request.

If a task needs a large catalog, retrieve tools before the model call.
ToolUniverse provides `tool_finder` and `find_tools_by_pattern`; a first pass can
select about ten tools and expose only those. This extra stage starts to pay off
when the fixed list grows past roughly 30 tools.

## Tool sets that map to this project's questions

`PROTEIN_CONTEXT_TOOLS` answers questions about the accessions behind the strongest
activation windows:

| Tool | Answers |
|---|---|
| `UniProt_get_recommended_name_by_accession` | what protein is this |
| `UniProt_get_function_by_accession` | what does it do |
| `UniProt_get_features_by_accession` | do the peak positions fall in an annotated feature |
| `UniProt_get_subcellular_location_by_accession` | is the latent tracking localization |
| `UniProt_get_ptm_processing_by_accession` | is it a modification site |
| `InterPro_get_entries_for_protein` | which domains |
| `InterPro_get_protein_domain_architecture` | domain order and boundaries |
| `Pfam_get_protein_annotations` | family assignment |
| `Reactome_map_uniprot_to_pathways` | pathway context |

`UniProt_get_features_by_accession` supports residue-level checks. "This latent
fires on kinases" is a protein-level association. "These peaks fall inside the
annotated ATP-binding regions" can be checked against the peak indices.

`MUTATION_EFFECT_TOOLS` supports the separate mutation-effect experiment described in
[03-sae-and-mutation-effects.md](03-sae-and-mutation-effects.md):
`UniProt_get_disease_variants_by_accession`, `MaveDB_search_score_sets`,
`MaveDB_get_variant_scores`, `ClinVar_search_variants`, `gnomad_get_constraint`.

Other candidates are `InterPro_get_residue_annotations` for residue-level records,
`alphafold_get_annotations` for structural context, `InterProScan_scan_sequence`
for sequences without accessions, and `EBI_msa_align` for conservation at the
firing site. None is exposed by default.

## Operational notes

- **Tool calls use live network services.** Six lookups for each of 1,000 latents
  means 6,000 requests to services such as EBI and UniProt. Use ToolUniverse's
  cache (`use_cache=True` on `run`, then `dump_cache()`) and limit the request rate.
- **Results are large.** One UniProt entry can exceed the useful context budget on
  its own; `_stringify` truncates at 8k characters. If truncation is cutting the
  informative part, prefer a narrower tool over a bigger limit.
- **Failures stay in the trace.** A missing accession or an HTTP 500 becomes an
  `is_error` result, so the model can try another accession. This also records
  which parts of the corpus could not be retrieved.
- **Keep the tool list ordered and stable.** Reordering changes the prompt prefix
  and defeats any provider-side caching; it also makes runs non-comparable.

## Extending

Add a tool to the relevant tuple in `config.py`. It should supply evidence for a
specific schema field; otherwise its definition consumes context without improving
the annotation.
