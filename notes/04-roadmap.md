# Roadmap

These are proposed steps, in dependency order. None is committed work.

## Now: replace the fixture with real SAE output

`LatentEvidenceStore` is the input boundary. Export per-residue activations as
`{latent_id: {protein_id: [act, ...]}}`, then load them with
`from_activation_dump`. No other module needs SAE-specific code.

Decide how to handle two sampling problems:

- **Corpus bias.** Whatever the proteins over-represent, the annotations will
  over-represent. Record the composition of the corpus alongside the annotations.
- **Example selection.** `build_latent_evidence` ranks proteins by their maximum
  activation, so its examples come from the extreme tail. Add an option to sample
  across activation deciles if the agent also needs examples near typical firing
  strength.

## Next: close the evaluation loop

The current pipeline does not score its annotations. Adapt the protocol from Bills
et al. to proteins:

1. Agent produces a label from the top-k windows.
2. A second call sees the label but not the activations and predicts activation on held-out
   sequences.
3. Correlate predicted against actual.

Use the correlation as an interpretability score for the latent. Run the same
procedure with random labels to estimate how much correlation comes from the
evaluation protocol itself.

Run three cheaper checks first:

- **Self-consistency.** Annotate each latent three times at temperature greater
  than zero. Record the disagreement rate.
- **Control latents.** Feed randomly-permuted activations. Anything not labelled
  `uninterpretable` is a calibration failure.
- **Human audit.** Grade 20 to 30 latents by hand and use the result as a reference.

## Then: the mutation-effect agent

Build a second configuration around `MUTATION_EFFECT_TOOLS`, with its own schema
and system prompt. The annotation agent must not see MaveDB results before it
predicts sensitivity.

Experimental order:
1. Annotate latents (no DMS access at annotation time).
2. Predict `mutation_effect.predicted_sensitivity` per latent.
3. Score against ProteinGym / MaveDB at the residues each latent fires on.

This implements experiment C from
[03-sae-and-mutation-effects.md](03-sae-and-mutation-effects.md).

## Engineering work for larger runs

- **Caching.** Call ToolUniverse with `use_cache=True` and persist the cache with
  `dump_cache()`. Many latents point to the same accessions.
- **Concurrency.** The example script annotates sequentially in a plain loop
  (`for latent_id in ...: agent.annotate(...)`). Latents are independent, so a
  bounded thread pool can run `annotate` calls concurrently. Set the bound from
  service rate limits and the slowest endpoint.
- **Checkpointing.** Write JSONL as you go (the example already does) and skip
  latents already present, so a 10k-latent run survives interruption.
- **Cost and latency.** Record tokens and wall time per latent before attempting a
  full sweep.

## Model questions

- **Which Qwen-3 variant?** Benchmark smaller models on the same latent set. Use
  `--model` to change the model without changing the experiment code.
- **Tool-calling reliability.** Track how often the model skips
  `get_latent_evidence` (the loop already nudges and records it) and how often JSON
  needs repair. Test prompt changes and a smaller tool surface against these rates.
- **Provider variance.** HF routes to whichever provider serves the model; support
  for `response_format` and tool calling differs between them. Pin `provider=` in
  `HuggingFaceChatClient` before comparing runs.
- **A reference model.** Annotate the same few dozen latents with a stronger model
  and compare the results. A new `ChatClient` implementation is enough; the agent
  loop does not need to change.

## Questions to answer before scaling up

- Does annotation quality depend more on the tool surface or on the model? Ablate
  by running with `--no-tooluniverse` (local evidence only) and comparing.
- How stable are annotations across two SAEs trained on the same activations? If
  unstable, individual latent claims are not publishable and the unit of analysis
  has to be something coarser.
- What fraction of latents are `uninterpretable`? If the pipeline never uses that
  label, its confidence calibration is suspect.
