# ComfyUI-Flux2-JiT

A practical ComfyUI custom-node implementation of **Just-in-Time (JiT)** for **Flux.2** sampling.

This repo implements the paper idea as a **Flux.2-focused ComfyUI integration** rather than a line-for-line port of the authors' Diffusers pipelines. The design preserves the official release's inference-time structure where that structure is useful, but moves the stateful sampling logic into a ComfyUI custom sampler so latent-state transitions are applied to the real sampler state instead of only to isolated model calls.

## Scope

Primary target:
- **Flux.2** in current ComfyUI

Current status:
- **Implemented:** Flux.2 text-to-image / image-generation path without ControlNet, reference latents, or masked denoising.
- **Not implemented yet:** Flux.1, ControlNet, Kontext/reference-latent paths, inpainting masks, multi-eval samplers such as Heun/DPM families.

## Why the implementation is split across a model patch and a sampler

A pure diffusion-model wrapper is not enough for JiT in ComfyUI, because the paper and official code both modify the latent state during stage transitions. In ComfyUI, if that logic only lives inside a diffusion-model wrapper, the external sampler state is unchanged, which breaks the intended DMF transition semantics.

This repo therefore uses two pieces:

1. **Model patch (`Flux2 JiT Apply`)**
   - Replaces dense Flux.2 diffusion-model forward passes with sparse-token forward passes plus interpolation.
   - This corresponds to the **SAG-ODE / augmented lifter** part of JiT.

2. **Custom sampler (`Flux2 JiT Sampler`)**
   - Owns the stage schedule, token activation, DMF transition, and Euler-style latent update loop.
   - This is where the latent state is actually updated before each evaluation when a stage transition happens.

That split is the lowest-risk way to stay faithful to JiT's control flow inside ComfyUI.

## Repository structure

```text
ComfyUI-Flux2-JiT/
├── __init__.py
├── nodes.py
├── LICENSE
├── README.md
├── docs/
│   └── IMPLEMENTATION_NOTES.md
└── flux2_jit/
    ├── config.py
    ├── interpolation.py
    ├── runtime.py
    ├── utils.py
    └── wrappers.py
```

## Installation

Clone or copy this repo into `ComfyUI/custom_nodes/`:

```bash
git clone <this-repo> ComfyUI/custom_nodes/ComfyUI-Flux2-JiT
```

Restart ComfyUI.

## Recommended workflow

Use ComfyUI's **custom sampling** path, not the standard `KSampler` path.

Recommended graph:

1. Load a **Flux.2** model.
2. Apply **Flux2 JiT Apply** to the model.
3. Build conditioning as usual.
4. Use **Flux2Scheduler** from ComfyUI core for sigma generation.
5. Use **Flux2 JiT Sampler** as the `SAMPLER` input to **SamplerCustom**.
6. Use an empty Flux.2 latent and decode as usual.

### Recommended settings

For the official presets:

- **4x preset**
  - `preset = default_4x`
  - `Flux2Scheduler steps = 18`

- **7x preset**
  - `preset = default_7x`
  - `Flux2Scheduler steps = 11`

These step counts are the closest match to the authors' released Flux.2 JiT code.

## Nodes

### Flux2 JiT Apply

Patches a Flux.2 model with JiT sparse-token inference.

Inputs:
- `model`: Flux.2 model
- `preset`: `default_4x`, `default_7x`, or `custom`
- `expected_total_steps`: used for validation/documentation; runtime sigma count still wins
- `stage_ratios`: comma-separated stage boundaries, default `0.4,0.65,1.0`
- `sparsity_ratios`: comma-separated active-token ratios, default `0.35,0.62,1.0`
- `use_checkerboard_init`
- `use_adaptive`
- `microflow_relax_steps`
- `blur_scale`
- `verbose`

Output:
- patched `MODEL`

### Flux2 JiT Sampler

Produces a `SAMPLER` object for use with ComfyUI `SamplerCustom`.

Output:
- `SAMPLER`

## Mapping from paper components and official-code components to this repo

### Paper components → repo modules

- **Nested token-subset chain / projectors**
  - `flux2_jit/runtime.py`
  - `flux2_jit/utils.py`

- **SAG-ODE / augmented lifter**
  - `flux2_jit/wrappers.py`
  - `flux2_jit/interpolation.py`

- **Interpolation operators `I_k` and `Φ_k`**
  - `flux2_jit/interpolation.py`

- **Importance-guided token activation (ITA)**
  - `flux2_jit/runtime.py`

- **Deterministic micro-flow (DMF)**
  - `flux2_jit/runtime.py`

- **JiT sampling loop**
  - `nodes.py` (`Flux2JiTSamplerImpl`)

### Official released Flux.2 JiT code → repo modules

- `pipeline_flux2_klein_JiT.py::set_params`
  - `flux2_jit/config.py`

- `pipeline_flux2_klein_JiT.py::_create_sparse_grid`
  - `flux2_jit/utils.py::create_sparse_grid`

- `pipeline_flux2_klein_JiT.py::_irregular_interpolation`
  - `flux2_jit/interpolation.py::irregular_interpolation`

- `pipeline_flux2_klein_JiT.py::_compute_importance_map`
  - `flux2_jit/runtime.py::_compute_importance_map`

- `pipeline_flux2_klein_JiT.py::_adaptive_densify`
  - `flux2_jit/runtime.py::_adaptive_densify`

- `pipeline_flux2_klein_JiT.py::_microflow_bridge`
  - `flux2_jit/runtime.py::_microflow_bridge`

- `pipeline_flux2_klein_JiT.py::__call__`
  - split between:
    - `nodes.py::Flux2JiTSamplerImpl.sample`
    - `flux2_jit/runtime.py`
    - `flux2_jit/wrappers.py`

## Explicit mismatches, ambiguities, and chosen resolutions

Detailed notes are in `docs/IMPLEMENTATION_NOTES.md`.

The most important ones:

1. **Paper vs official code: DMF**
   - The paper describes a finite-time hitting ODE.
   - This repo applies progressive relaxation across `microflow_relax_steps` denoising calls, with per-call blend weight `1 / remaining_steps`.
   - This makes the `microflow_relax_steps` control semantically active instead of only scaling a one-shot blend.

2. **Paper vs ComfyUI sampling semantics**
   - The paper is written in a direct generative-ODE form.
   - ComfyUI Flux.2 uses its own flow-model denoiser/sampler semantics.
   - This repo resolves that by using the ComfyUI identity:
     - `velocity = (x - denoised) / sigma`
     - `x0_pred = x - sigma * velocity`
   - This matches ComfyUI Flux.2 behavior and is closer to the released Flux.2 code than forcing the paper notation directly.

3. **Official Flux.2 diffusers pipeline vs ComfyUI integration point**
   - The released code owns the whole sampling loop.
   - In ComfyUI, the cleanest faithful port is a **custom sampler + sparse model wrapper**, not a single pipeline file.

4. **Scheduler mismatch**
   - The released Flux.2 code uses Flux.2's native schedule.
   - In ComfyUI, the closest equivalent is the built-in **Flux2Scheduler** node.
   - This repo therefore expects that scheduler in the recommended workflow.

5. **Unsupported conditioning modes**
   - The released Flux.2 JiT code also has image-conditioned handling.
   - This repo currently falls back to dense behavior or refuses unsupported cases instead of pretending those branches are implemented.

## Assumptions

- ComfyUI's current Flux.2 implementation retains:
  - `diffusion_model.process_img(...)`
  - `diffusion_model.forward_orig(...)`
  - `diffusion_model.patch_size`
- Flux.2 latent patching/layout stays compatible with the current ComfyUI master the repo was developed against.
- The intended sampler path is **Euler-like one-eval-per-step custom sampling**, using `SamplerCustom` and `Flux2Scheduler`.

## Limitations

- Only **Flux.2** is implemented.
- Only the **custom sampling path** is supported.
- No ControlNet, no reference latents, no masked denoising.
- Not validated for multi-evaluation samplers such as Heun/DPM++ variants.
- Not intended to be stacked with unrelated `DIFFUSION_MODEL` wrappers that also replace Flux.2 forward behavior.
- Batch behavior assumes a shared active-token set per sampled latent batch.

## Unresolved uncertainties

- The paper's notation and ComfyUI Flux.2's sigma/denoised conventions are not written in the same parameterization. The chosen mapping is consistent with ComfyUI's model-sampling code and the authors' released Flux.2 implementation, but it is still an integration inference, not a direct statement from the paper.
- The released code's DMF implementation is weaker than the paper's stated finite-time ODE. This repo intentionally preserves that released behavior.
- The official code's initial sparse-grid construction does a random fill/drop step to meet the exact token budget. This repo preserves that behavior, which means the exact initial token set is not deterministic across runs unless the torch RNG state is controlled before graph execution.
