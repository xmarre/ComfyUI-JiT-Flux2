from __future__ import annotations

import copy
from dataclasses import replace

import torch

import comfy.model_patcher
import comfy.patcher_extension
import comfy.samplers
from comfy.k_diffusion import utils as k_utils

from .flux2_jit.config import DEFAULT_4X_STEPS, DEFAULT_7X_STEPS, config_from_inputs
from .flux2_jit.runtime import JiTRuntime
from .flux2_jit.utils import is_flux2_model, log_info
from .flux2_jit.wrappers import JIT_CONFIG_KEY, JIT_RUNTIME_KEY, flux2_jit_diffusion_model_wrapper


class Flux2JiTApply:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "preset": (("default_4x", "default_7x", "custom"), {"default": "default_4x"}),
                "expected_total_steps": ("INT", {"default": DEFAULT_4X_STEPS, "min": 1, "max": 10000}),
                "stage_ratios": ("STRING", {"default": "0.4,0.65,1.0", "multiline": False}),
                "sparsity_ratios": ("STRING", {"default": "0.35,0.62,1.0", "multiline": False}),
                "use_checkerboard_init": ("BOOLEAN", {"default": True}),
                "use_adaptive": ("BOOLEAN", {"default": True}),
                "microflow_relax_steps": ("INT", {"default": 3, "min": 0, "max": 64}),
                "blur_scale": ("FLOAT", {"default": 0.4, "min": 0.05, "max": 4.0, "step": 0.01}),
                "verbose": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "sampling/custom_sampling/flux2_jit"

    def apply(
        self,
        model,
        preset,
        expected_total_steps,
        stage_ratios,
        sparsity_ratios,
        use_checkerboard_init,
        use_adaptive,
        microflow_relax_steps,
        blur_scale,
        verbose,
    ):
        if not is_flux2_model(model):
            raise ValueError("Flux2JiTApply currently supports ComfyUI Flux.2 models only")

        config = config_from_inputs(
            preset=preset,
            expected_total_steps=expected_total_steps,
            stage_ratios_csv=stage_ratios,
            sparsity_ratios_csv=sparsity_ratios,
            use_checkerboard_init=use_checkerboard_init,
            use_adaptive=use_adaptive,
            microflow_relax_steps=microflow_relax_steps,
            blur_scale=blur_scale,
            verbose=verbose,
        )

        patched = model.clone()
        patched.model_options.setdefault("transformer_options", {})[JIT_CONFIG_KEY] = config
        patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "flux2_jit", flux2_jit_diffusion_model_wrapper)
        return (patched,)


class Flux2JiTSamplerImpl(comfy.samplers.Sampler):
    def sample(self, model_wrap, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None, disable_pbar=False):
        if denoise_mask is not None:
            raise ValueError("Flux2JiTSampler does not support masked/inpaint denoising")

        model_options = extra_args["model_options"]
        transformer_options = model_options.setdefault("transformer_options", {})
        config = transformer_options.get(JIT_CONFIG_KEY)
        if config is None:
            raise ValueError("Flux2JiTSampler requires a model patched by Flux2JiTApply")
        if not is_flux2_model(model_wrap.model_patcher):
            raise ValueError("Flux2JiTSampler currently supports ComfyUI Flux.2 models only")

        if config.expected_total_steps != (len(sigmas) - 1):
            log_info(config.verbose, f"Configured steps={config.expected_total_steps}, runtime sigmas={len(sigmas) - 1}; using runtime sigmas")

        x = model_wrap.inner_model.model_sampling.noise_scaling(sigmas[0], noise, latent_image, self.max_denoise(model_wrap, sigmas))
        runtime = JiTRuntime(
            config=config,
            total_steps=len(sigmas) - 1,
            sigmas=sigmas,
            global_noise_image=torch.randn_like(x),
        )
        transformer_options[JIT_RUNTIME_KEY] = runtime

        total_steps = len(sigmas) - 1
        s_in = x.new_ones([x.shape[0]])
        try:
            for i in range(total_steps):
                sigma = sigmas[i]
                sigma_vec = sigma * s_in
                x = runtime.maybe_apply_stage_transition(model_wrap.inner_model.diffusion_model, x, i, sigma_vec)
                denoised = model_wrap(x, sigma_vec, **extra_args)
                velocity = (x - denoised) / k_utils.append_dims(sigma_vec, x.ndim)
                runtime.last_x_image = x.detach().clone()
                runtime.last_velocity_image = velocity.detach().clone()
                runtime.last_sigma = sigma_vec.detach().clone()
                if callback is not None:
                    callback(i, denoised, x, total_steps)
                dt = sigmas[i + 1] - sigma
                x = x + velocity * dt
            return model_wrap.inner_model.model_sampling.inverse_noise_scaling(sigmas[-1], x)
        finally:
            transformer_options.pop(JIT_RUNTIME_KEY, None)


class Flux2JiTSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "build"
    CATEGORY = "sampling/custom_sampling/flux2_jit"

    def build(self):
        return (Flux2JiTSamplerImpl(),)


NODE_CLASS_MAPPINGS = {
    "Flux2JiTApply": Flux2JiTApply,
    "Flux2JiTSampler": Flux2JiTSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux2JiTApply": "Flux2 JiT Apply",
    "Flux2JiTSampler": "Flux2 JiT Sampler",
}
