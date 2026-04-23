# Implementation Notes

## What the paper explicitly specifies

- JiT has two main components:
  - SAG-ODE
  - DMF
- A nested active-token schedule is used.
- New token activation is driven by a variance-based importance map.
- Interpolation is nearest-neighbor plus masked Gaussian blur.
- The official Flux.2 release is an inference-time method with no training path required.

## What the official released code explicitly implements

- Flux.2 support is inference-only.
- Initial anchors are built with a checkerboard-plus-boundary style grid and then randomly adjusted to match the target budget.
- ITA is implemented as top-k selection over normalized local variance.
- DMF is implemented as a single weighted blend, not as a numerically integrated finite-time hitting ODE.
- Flux.2 uses its native timestep schedule rather than the beta-warped schedule used for Flux.1 in the released repo.

## What had to be inferred for ComfyUI

- Where JiT should live in ComfyUI's execution graph.
- How to make stage transitions update the actual sampler state rather than only an internal temporary tensor.
- How to translate the released Diffusers pipeline control flow into ComfyUI's `MODEL` + `SAMPLER` + `SIGMAS` abstraction.
- How to recover the CFG-combined velocity field in ComfyUI sampling semantics.

## Pragmatic implementation choices in this repo

- JiT is split into:
  - a sparse diffusion-model wrapper
  - a custom sampler that owns stage transitions
- The sampler path is limited to Euler-style custom sampling because that is the lowest-risk match to the released JiT control flow.
- Unsupported branches are rejected or allowed to fall back, instead of being replaced with fake partial support.

## Root correctness invariant

The sparse-token forward path and the stage-transition path must both operate on the **same evolving latent state**. Any design that computes sparse outputs on an internal surrogate state while leaving the external sampler latent unchanged violates JiT's control-flow invariant.

## Competing integration hypothesis that was rejected

### Rejected approach
Implement JiT entirely as a `DIFFUSION_MODEL` wrapper and keep ComfyUI's stock samplers unchanged.

### Why it was rejected
That approach can sparsify the model call, but it cannot faithfully apply the DMF stage transition to the sampler's actual latent state. The sampler would continue evolving the old state, while the wrapper would be evaluating a modified internal state. That is a structural mismatch, not a small approximation.

## Remaining risk areas

- Upstream ComfyUI changes to Flux.2 tokenization or forward signatures.
- Differences between official JiT timing assumptions and arbitrary sigma schedules.
- Unsupported branches such as ControlNet or reference-latent conditioning.
