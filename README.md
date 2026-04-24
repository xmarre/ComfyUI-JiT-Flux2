# ComfyUI-JiT-Flux2

An unofficial ComfyUI implementation of **Just-in-Time (JiT)** for **Flux.2**.

JiT is a **training-free spatial acceleration** method for diffusion transformers. Instead of evaluating the full latent token grid at every denoising step, it evaluates a sparse subset of spatial anchor tokens, lifts/interpolates the sparse velocity back to the full latent grid, and progressively activates more tokens as sampling proceeds.

This repository adapts the paper idea to current ComfyUI Flux.2 workflows. It is a practical ComfyUI port, not an official release by the paper authors and not a byte-for-byte reproduction of the original FLUX.1-dev setup.

---

## Paper and credit

This repository is based on:

> **Wenhao Sun, Ji Li, and Zhaoqiang Liu**  
> **Just-in-Time: Training-Free Spatial Acceleration for Diffusion Transformers**  
> arXiv:2603.10744, 2026

- **Paper:** [arXiv:2603.10744](https://arxiv.org/abs/2603.10744)
- **Project page:** [JiT project page](https://wenhao-sun77.github.io/JiT/)

```bibtex
@article{sun2026jit,
  title   = {Just-in-Time: Training-Free Spatial Acceleration for Diffusion Transformers},
  author  = {Sun, Wenhao and Li, Ji and Liu, Zhaoqiang},
  journal = {arXiv preprint arXiv:2603.10744},
  year    = {2026}
}
```

---

## Method summary

JiT uses a coarse-to-fine token schedule:

1. Start from a sparse subset of spatial anchor tokens.
2. Run the transformer only on active tokens.
3. Lift the sparse velocity back to the full latent grid.
4. Activate more tokens at stage boundaries.
5. Use a deterministic micro-flow target so newly activated tokens enter with coherent structure and the correct noise level.

Implemented paper components:

- **SAG-ODE:** sparse-token model evaluation plus full-grid velocity lifting.
- **ITA:** importance-guided token activation from local velocity variance.
- **DMF:** stage-transition target construction for newly activated tokens.
- **Interpolation:** nearest-neighbor expansion plus controlled Gaussian blur while preserving exact anchor values.

The paper evaluates FLUX.1-dev. This repo targets Flux.2, so the defaults are Flux.2-oriented and intentionally more conservative than the paper's FLUX.1-dev sparsities.

---

## Installation

Clone into `ComfyUI/custom_nodes/`:

```bash
git clone https://github.com/xmarre/ComfyUI-JiT-Flux2.git ComfyUI/custom_nodes/ComfyUI-JiT-Flux2
```

Restart ComfyUI after installation.

---

## Recommended workflow

Use ComfyUI's **custom sampling** path.

1. Load a **Flux.2** model.
2. Apply **Flux2 JiT Apply** to the model.
3. Build conditioning as usual.
4. Generate sigmas with **Flux2 JiT Scheduler**.
5. Feed **Flux2 JiT Sampler** into **SamplerCustom**.
6. Start from an empty Flux.2 latent and decode normally.

The new scheduler is based on ComfyUI core Flux2Scheduler: it takes `steps`, image-space `width`, and image-space `height`, computes the Flux.2 empirical sequence-length shift from the image dimensions, and returns `SIGMAS` for SamplerCustom.

---

## Recommended presets

| Preset | Scheduler steps | Intended tier | Stage sparsities |
|---|---:|---|---|
| `default_4x` | 18 | conservative Flux.2 ~4x tier | `45% -> 70% -> 100%` |
| `default_7x` | 11 | conservative Flux.2 ~7x tier | `40% -> 66% -> 100%` |

These are **Flux.2-tuned conservative defaults**, not guaranteed benchmark-equivalent reproductions of the paper's FLUX.1-dev numbers. The paper-reported FLUX.1-dev sparsities were more aggressive (`35% -> 62% -> 100%` and `32% -> 60% -> 100%`). Flux.2 currently benefits from higher early and mid-stage token coverage, especially for text, small details, and anatomy.

---

## Nodes

### `Flux2 JiT Apply`

Patches a Flux.2 model with JiT sparse-token inference behavior.

Inputs:

- `model`: Flux.2 model.
- `preset`: `default_4x`, `default_7x`, or `custom`.
- `expected_total_steps`: expected denoising step count. Runtime sigma count still controls execution.
- `stage_ratios`: comma-separated stage boundaries. Default: `0.4,0.65,1.0`.
- `sparsity_ratios`: comma-separated active-token ratios. Default preset uses `0.45,0.70,1.0`.
- `use_checkerboard_init`: enabled by default. Uses paper-style strided/checkerboard initialization with boundary coverage for the first sparse stage.
- `use_adaptive`: enables importance-guided adaptive token activation.
- `microflow_relax_steps`: retained as a compatibility/control input, but stage activation commits only after the DMF target is reached.
- `blur_scale`: interpolation blur strength.
- `verbose`: runtime logging.

Output: patched `MODEL`.

### `Flux2 JiT Scheduler`

Produces `SIGMAS` for `SamplerCustom`.

Inputs:

- `steps`: number of denoising steps. Start with `18` for `default_4x` and `11` for `default_7x`.
- `width`, `height`: final image-space dimensions passed to Flux.2. Do not pass latent-token sizes. Non-16-aligned values are quantized internally to the latent token grid before computing the Flux.2 empirical sequence-length shift.
- `schedule`: `flux2` or `jit_beta`.
  - `flux2` keeps the Flux2Scheduler-style shifted linear schedule and is the default.
  - `jit_beta` applies a JiT paper-style beta warp before the Flux.2 empirical shift. Treat it as experimental until visually validated on Flux.2.
- `beta_alpha`, `beta_beta`, `beta_resolution`: parameters for `jit_beta` mode. Defaults are based on the paper's reported beta schedule.

### `Flux2 JiT Sampler`

Produces a `SAMPLER` object for use with ComfyUI `SamplerCustom`.

---

## What changed from the initial Flux.2 port

The first implementation exposed two avoidable quality degradation paths:

- freshly sampled transition noise for newly activated tokens instead of reusing the sampler noise;
- committing newly activated tokens to `current_indices` before their DMF target was reached.

Those were fixed. Remaining quality differences from non-JiT are still expected because JiT changes the actual ODE trajectory, but this repo now also avoids another bad default: the non-checkerboard sparse-grid path no longer selects the whole grid and randomly drops tokens at common sparsity ratios. Sparse-grid adjustment is deterministic and boundary-aware, and the default selector is checkerboard/strided initialization again.

---

## Scope and limitations

Implemented:

- Flux.2 text-to-image / image-generation path.
- JiT sparse-token model patching.
- JiT custom sampler path for stage scheduling and latent-state transitions.
- Flux2Scheduler-derived JiT scheduler node.

Not implemented:

- Flux.1 support.
- ControlNet paths.
- masked denoising / inpainting.
- samplers that require multiple model evaluations per step, such as some Heun / DPM-family paths.

The intended execution path is:

- **Flux.2**
- **Flux2 JiT Scheduler** or ComfyUI core **Flux2Scheduler**
- **SamplerCustom**
- **Euler-like one-eval-per-step custom sampling**

---

## Repository structure

```text
ComfyUI-JiT-Flux2/
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
    ├── scheduler.py
    ├── utils.py
    └── wrappers.py
```

---

## Method-to-code mapping

- **Nested token-subset chain / projectors:** `flux2_jit/runtime.py`, `flux2_jit/utils.py`
- **SAG-ODE / augmented lifter:** `flux2_jit/wrappers.py`, `flux2_jit/interpolation.py`
- **Interpolation operators `I_k` and `Φ_k`:** `flux2_jit/interpolation.py`
- **Importance-guided token activation:** `flux2_jit/runtime.py`
- **Deterministic micro-flow:** `flux2_jit/runtime.py`
- **Flux.2 JiT scheduler:** `flux2_jit/scheduler.py`
- **ComfyUI nodes:** `nodes.py`

---

## Acknowledgments

Full credit for the JiT method belongs to the paper authors:

- **Wenhao Sun**
- **Ji Li**
- **Zhaoqiang Liu**

This repository is an unofficial ComfyUI adaptation of their method for Flux.2 workflows.
