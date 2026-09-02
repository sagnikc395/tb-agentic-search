# SAE latents, annotation, and mutation effect prediction

## The setup

A protein language model such as ESM-2, ESM-C, or ProtT5 produces a vector for
each residue. Its dimensions are dense and polysemantic: no single dimension has
one stable meaning. A sparse autoencoder trained on those activations,

```
z = ReLU(W_enc (h - b_dec) + b_enc)     # sparse, overcomplete: dim(z) >> dim(h)
ĥ = W_dec z + b_dec
```

produces a larger set of latents, each active on a small fraction of residues. The
hypothesis is that these sparse latents correspond more closely to features used by
the model than the original dimensions do.

Testing that hypothesis requires descriptions of the latents. An SAE may contain
10,000 to 100,000 of them, too many to inspect one by one.

## Why "annotation" is harder than it sounds

Showing a model the top-activating sequences and asking for a label leaves three
problems:

1. **The model will always find a pattern.** Given any set of protein fragments, an
   LLM produces a fluent explanation. A label with no falsification condition is not
   a finding. The schema therefore requires `disconfirming_evidence`.

2. **Top-k activations are a biased sample.** They over-represent whatever the
   corpus over-represents. A latent that "fires on kinases" may be tracking the most
   common fold in the dataset rather than a kinase-specific feature. The schema
   requires `rejected_hypotheses`, and the prompt asks for a site the proposed label
   does not explain.

3. **Protein-level facts do not license residue-level claims.** "These are all
   kinases" is guilt by association. "These peaks fall inside the annotated
   ATP-binding region at positions X, Y, and Z" is a positional claim that can be checked against
   the peak index. `UniProt_get_features_by_accession` and
   `InterPro_get_residue_annotations` provide the records needed for that check.

Bills et al. (2023), *Language models can explain neurons in language models*,
first generate an explanation and then use it to predict activations. Comparing
those predictions with observed activations gives the explanation a score. This
repository does not yet implement that step; see [04-roadmap.md](04-roadmap.md).

## Mutation effect prediction: why latents are interesting here

Zero-shot variant effect prediction from a PLM is usually the masked-marginal
score,

```
score(wt -> mut at position i) = log p(mut | context) - log p(wt | context)
```

and is evaluated against deep mutational scanning data from ProteinGym or MaveDB.
The score alone does not identify which internal features changed.

SAE latents may provide that decomposition. If a few annotated latents account for
the score change, the explanation can name them: "the substitution suppresses a
latent associated with the catalytic-site configuration," for example.

The repository could support three experiments.

**A. Use latents to explain an existing score.** Compute the variant score as usual,
then attribute its change across latents that differ between the wild type and the
mutant. The agent supplies a label for each contributing latent. This experiment
does not change the predictor.

**B. Use latents as predictor inputs.** Fit a small model on latent activations at
the mutated position and compare it with the masked-marginal baseline on
ProteinGym. The result measures how much predictive signal the sparse features
retain.

**C. Test annotation-guided priors.** If a latent has a high-confidence
`active_or_binding_site` label, substitutions that suppress it should have larger
effects. The `mutation_effect.predicted_sensitivity` field records this prediction
before the agent sees DMS data.

Direction C produces a direct test: correlate `predicted_sensitivity` with the
measured DMS effects at residues where each latent fires.

## A separate mutation-effect agent

`MUTATION_EFFECT_TOOLS` contains the tools needed to ask whether substitutions at
a latent's firing sites have measured effects:

| Tool | Role |
|---|---|
| `UniProt_get_disease_variants_by_accession` | whether curated pathogenic variants land on the firing sites |
| `MaveDB_search_score_sets` / `MaveDB_get_variant_scores` | experimental DMS measurements for these proteins |
| `MaveDB_get_effect_matrix` | parses HGVS into (reference, position, alternate) and keeps single substitutions |
| `ClinVar_search_variants` | clinical significance of substitutions at the site |
| `gnomad_get_constraint` | population-level depletion of variation |

Withhold DMS data during annotation. Let the annotation agent predict sensitivity
from structure and function, then let a second process compare that prediction with
MaveDB or ProteinGym. If the first agent can retrieve DMS measurements, the test
measures retrieval rather than prediction.

## Experimental limits

- **SAE latents are not stable across training runs.** Two SAEs on the same
  activations give different bases. Any claim about "the catalytic site latent"
  needs a cross-run reproducibility check. Structured annotations make those runs
  easier to compare.
- **Feature splitting.** Wider SAEs split one concept into several narrower ones.
  The same annotation may legitimately apply to multiple latents at higher width.
- **Reconstruction is not interpretability.** A better L0/reconstruction tradeoff does not
  imply more interpretable latents; measure them separately.
- **Annotation is an LLM output, not ground truth.** Report inter-run agreement and
  a human-audited subset before treating annotations as labels in a downstream claim.

## References

- Bills et al. (2023), *Language models can explain neurons in language models*
- Cunningham et al. (2023), *Sparse autoencoders find highly interpretable features*
- Bricken et al. (2023), *Towards monosemanticity* (Anthropic)
- Simon & Zou (2024), *InterPLM: discovering interpretable features in PLMs via SAEs*
- Meier et al. (2021), *Language models enable zero-shot prediction of mutation effects*
- Notin et al. (2023), *ProteinGym*
