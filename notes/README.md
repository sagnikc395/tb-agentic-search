# Notes

These notes explain the design and planned experiments. They are ordered, but each
document can be read on its own.

| | |
|---|---|
| [01: Agent foundations](01-agent-foundations.md) | The agent loop, its three main design decisions, and project-specific failure cases. |
| [02: ToolUniverse](02-tooluniverse.md) | Which tools to expose and the cost of live database calls. |
| [03: SAE latents and mutation effects](03-sae-and-mutation-effects.md) | Limits of top-activation labels and three experiments for variant-effect prediction. |
| [04: Roadmap](04-roadmap.md) | Proposed work in dependency order. |
| [05: Codebase map and config](05-codebase-and-config.md) | File ownership, configuration, previous cleanup, conventions, and known risks. |
| [06: Repository deep dive](06-repository-deep-dive.md) | Current end-to-end behavior, runnable-check results, limitations, risks, and implementation priorities. |

The code these notes describe is in `latent_annotation/`, with a runnable demo in
`examples/`. See the top-level [README](../README.md) for how to run it.
