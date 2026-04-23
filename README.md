# ComfyUI-JiT-Flux2

An unofficial ComfyUI implementation of **Just-in-Time (JiT)** for **Flux.2**.

JiT is a **training-free spatial acceleration** method for diffusion transformers. Instead of evaluating the full latent token grid at every denoising step, it computes the model on a sparse subset of **anchor tokens** in earlier stages, interpolates the missing regions, and progressively activates more tokens as the sample develops.

This repository adapts that idea to **current ComfyUI Flux.2 workflows**. It is a practical, reviewable **ComfyUI port**, not the authors’ original pipeline and not a line-for-line reproduction of the paper’s FLUX.1-dev setup.

---

## Paper and credit

This repository is based on:

> **Wenhao Sun, Ji Li, and Zhaoqiang Liu**  
> **Just-in-Time: Training-Free Spatial Acceleration for Diffusion Transformers**  
> arXiv:2603.10744, 2026

- **Paper:** [arXiv:2603.10744](https://arxiv.org/abs/2603.10744)
- **Project page:** [JiT project page](https://wenhao-sun77.github.io/JiT/)

If you use this repository, please also cite the paper.

### BibTeX

```bibtex
@article{sun2026jit,
  title   = {Just-in-Time: Training-Free Spatial Acceleration for Diffusion Transformers},
  author  = {Sun, Wenhao and Li, Ji and Liu, Zhaoqiang},
  journal = {arXiv preprint arXiv:2603.10744},
  year    = {2026}
}
```

---

## What JiT does

The paper’s central observation is that diffusion transformers do **not** need uniform spatial compute throughout the entire denoising trajectory. Global structure tends to emerge early, while fine details are refined later. JiT exploits that by using a **coarse-to-fine token schedule**:

1. Start from a sparse subset of spatial tokens.
2. Run the transformer only on those active tokens.
3. Extrapolate a full latent update from the sparse result.
4. Gradually activate more tokens in regions that appear most important.
5. Transition newly activated tokens in a way that preserves both structure and the correct noise level.

In paper terms, the main pieces are:

- **SAG-ODE (Spatially Approximated Generative ODE)**  
  Computes the velocity field on sparse anchor tokens and lifts it back to the full token space.

- **ITA (Importance-Guided Token Activation)**  
  Selects which inactive tokens to activate next by looking at local variation in the predicted velocity field.

- **DMF (Deterministic Micro-Flow)**  
  Handles stage transitions so newly activated tokens are injected with a structurally coherent and statistically consistent target state, rather than appearing abruptly.

The paper appendix further describes the interpolation operators used for JiT as a combination of:

- nearest-neighbor interpolation,
- controlled Gaussian blur,
- masked composition that preserves exact anchor values.

---

## What the paper reports

The JiT paper evaluates the method on **FLUX.1-dev**, not Flux.2. Its reported findings are important context for this repo, but they should not be read as direct benchmark claims for this ComfyUI Flux.2 port.

Paper-reported results include:

- a **~4× tier** using **18 NFEs**
- a **~7× tier** using **11 NFEs**
- strong quantitative results relative to both low-NFE vanilla FLUX.1-dev and other acceleration baselines
- blind user-study preference wins against all compared baselines
- ablations showing that removing the spatial approximation, dynamic token activation, or DMF target hurts quality

The paper’s reported 3-stage schedules are:

- **~4×:** `35% -> 62% -> 100%` active tokens across **18 NFEs**
- **~7×:** `32% -> 60% -> 100%` active tokens across **11 NFEs**

This repository uses those results and design ideas as the conceptual reference point, while adapting the implementation to ComfyUI and Flux.2.

---

## What this repository implements

### Primary target

- **Flux.2** in current ComfyUI

### Currently implemented

- Flux.2 text-to-image / image-generation path
- JiT sparse-token model patching
- JiT custom sampler path for stage scheduling and latent-state transitions
- preset schedules corresponding to the paper’s approximate **~4×** and **~7×** tiers

### Not implemented yet

- Flux.1 support
- ControlNet paths
- masked denoising / inpainting
- samplers that require multiple model evaluations per step (for example some Heun / DPM-family paths)

---

## Important scope note

This is an **independent ComfyUI adaptation** of JiT for **Flux.2**.

It is **not**:

- an official release by the paper authors
- a claim of exact paper-level benchmark reproduction
- a byte-for-byte port of the original FLUX.1-dev Diffusers pipeline

Where the paper describes algorithmic intent, this repo tries to preserve that intent. Where ComfyUI and Flux.2 impose different integration constraints, this repo chooses the smallest practical adaptation that keeps the control flow coherent.

---

## Why this repo is split into a model patch and a sampler

A pure diffusion-model wrapper is not sufficient for JiT inside ComfyUI.

The reason is that JiT is not only about sparse forward passes. It also changes the **latent state** when stages transition and new tokens become active. If that logic lived only inside a model wrapper, ComfyUI’s outer sampler state would not actually see the intended transition.

This repository therefore splits the implementation into two parts:

### 1. `Flux2 JiT Apply`

A model patch that replaces dense Flux.2 diffusion-model forward passes with JiT-style sparse-token forward passes plus interpolation.

This covers the **sparse evaluation / lifted full-space update** part of the method.

### 2. `Flux2 JiT Sampler`

A custom sampler that owns:

- the stage schedule
- token activation
- stage-transition handling
- latent updates during sampling

This is where JiT’s transition logic is applied to the **real sampler state**, which is the key requirement for a faithful ComfyUI integration.

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
    ├── utils.py
    └── wrappers.py
```

---

## Installation

Clone into `ComfyUI/custom_nodes/`:

```bash
git clone https://github.com/xmarre/ComfyUI-JiT-Flux2.git ComfyUI/custom_nodes/ComfyUI-JiT-Flux2
```

Restart ComfyUI after installation.

---

## Recommended workflow

Use ComfyUI’s **custom sampling** path, not the standard `KSampler` path.

### Recommended graph

1. Load a **Flux.2** model.
2. Apply **Flux2 JiT Apply** to the model.
3. Build positive / negative conditioning as usual.
4. Generate sigmas using **Flux2Scheduler** from ComfyUI core.
5. Feed **Flux2 JiT Sampler** into **SamplerCustom**.
6. Start from an empty Flux.2 latent and decode normally.

### Recommended presets

| Preset | Flux2Scheduler steps | Intended tier | Stage sparsities |
|---|---:|---|---|
| `default_4x` | 18 | paper-like ~4× tier | `35% -> 62% -> 100%` |
| `default_7x` | 11 | paper-like ~7× tier | `32% -> 60% -> 100%` |

These are **reference presets**, not guaranteed benchmark-equivalent reproductions of the paper’s FLUX.1-dev numbers.

---

## Nodes

## `Flux2 JiT Apply`

Patches a Flux.2 model with JiT sparse-token inference behavior.

### Inputs

- `model`  
  Flux.2 model

- `preset`  
  `default_4x`, `default_7x`, or `custom`

- `expected_total_steps`  
  Expected total denoising steps. Used for validation / documentation. Runtime sigma count still determines actual execution.

- `stage_ratios`  
  Comma-separated stage boundaries. Default: `0.4,0.65,1.0`

- `sparsity_ratios`  
  Comma-separated active-token ratios. Default: `0.35,0.62,1.0`

- `use_checkerboard_init`  
  Enable sparse-grid style initialization for the first stage

- `use_adaptive`  
  Enable importance-guided adaptive token activation

- `microflow_relax_steps`  
  Controls how stage-transition relaxation is spread across denoising calls

- `blur_scale`  
  Controls interpolation blur strength

- `verbose`  
  Enable runtime logging

### Output

- patched `MODEL`

---

## `Flux2 JiT Sampler`

Produces a `SAMPLER` object for use with ComfyUI `SamplerCustom`.

### Output

- `SAMPLER`

---

## Method-to-code mapping

### Paper concepts -> repo modules

- **Nested token-subset chain / projectors**  
  `flux2_jit/runtime.py`  
  `flux2_jit/utils.py`

- **SAG-ODE / augmented lifter**  
  `flux2_jit/wrappers.py`  
  `flux2_jit/interpolation.py`

- **Interpolation operators `I_k` and `Φ_k`**  
  `flux2_jit/interpolation.py`

- **Importance-guided token activation (ITA)**  
  `flux2_jit/runtime.py`

- **Deterministic micro-flow (DMF)**  
  `flux2_jit/runtime.py`

- **JiT sampling loop**  
  `nodes.py` (`Flux2JiTSamplerImpl`)

---

## Key implementation notes

The most important adaptation choices are:

### 1. Paper benchmark target vs this repo target

The paper benchmarks **FLUX.1-dev**. This repo targets **Flux.2 in ComfyUI**.

That means this repository should be understood as a **method port**, not as a claim that the exact paper benchmark setup has been reproduced.

### 2. ComfyUI integration differs from a monolithic pipeline

The paper and reference-style implementations can own the whole sampling loop directly.

In ComfyUI, the cleanest faithful port is a **custom sampler plus a sparse model wrapper**, because JiT needs to modify both model evaluation behavior and latent-state transitions.

### 3. Sampling semantics must match ComfyUI’s Flux.2 path

The paper is written in continuous ODE language, while ComfyUI Flux.2 operates through its own model/sampler interfaces. This repo therefore maps the JiT idea onto ComfyUI’s actual Flux.2 sampling semantics rather than trying to force the paper’s notation directly into an incompatible hook point.

### 4. Unsupported branches are not faked

Some modes discussed in paper-adjacent or reference implementations are not implemented here yet. This repo prefers to stay explicit about unsupported paths rather than silently falling back to misleading approximations.

See `docs/IMPLEMENTATION_NOTES.md` for more detail.

---

## Assumptions

This repository currently assumes ComfyUI’s Flux.2 implementation still exposes the model internals needed by the patch, including expected latent patch/layout behavior and the relevant diffusion-model call path.

The intended execution path is:

- **Flux.2**
- **Flux2Scheduler**
- **SamplerCustom**
- **Euler-like one-eval-per-step custom sampling**

---

## Limitations

- Flux.2 only
- custom sampling path only
- no ControlNet
- no masked denoising / inpainting
- not validated for multi-eval samplers such as Heun or DPM++ families
- not intended to be stacked with unrelated wrappers that also replace Flux.2 diffusion-model forward behavior
- batch behavior assumes a shared active-token set per latent batch

---

## Acknowledgments

Full credit for the JiT method belongs to the paper authors:

- **Wenhao Sun**
- **Ji Li**
- **Zhaoqiang Liu**

This repository is an unofficial ComfyUI adaptation of their method for Flux.2 workflows.

---

## Citation

If this repository is useful in your work, please cite the JiT paper:

```bibtex
@article{sun2026jit,
  title   = {Just-in-Time: Training-Free Spatial Acceleration for Diffusion Transformers},
  author  = {Sun, Wenhao and Li, Ji and Liu, Zhaoqiang},
  journal = {arXiv preprint arXiv:2603.10744},
  year    = {2026}
}
```
