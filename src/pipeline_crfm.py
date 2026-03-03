"""
Custom diffusers pipeline for CRFM (Control Rectified Flow Matching).

This pipeline supports mask-conditioned image generation using the MaskDit_sd3_5
transformer built on top of Stable Diffusion 3.5.

Usage (after converting a checkpoint with ``scripts/convert_checkpoint.py``):

    >>> from src.pipeline_crfm import CRFMPipeline
    >>> pipe = CRFMPipeline.from_pretrained("path/to/converted_pipeline")
    >>> pipe = pipe.to("cuda")
"""

from typing import Any, Dict, List, Optional, Union

import torch
from diffusers import (
    AutoencoderKL,
    DiffusionPipeline,
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.stable_diffusion_3.pipeline_output import (
    StableDiffusion3PipelineOutput,
)

from src.models.sd3_mmdit import MaskDit_sd3_5
from src.utils.utils import (
    _prepare_latents,
    calculate_shift,
    encode_images,
    retrieve_timesteps,
)


class CRFMPipeline(DiffusionPipeline):
    """Diffusers-compatible pipeline for CRFM mask-conditioned generation.

    The pipeline wraps:
    * A :class:`MaskDit_sd3_5` transformer (the core generative model).
    * A VAE encoder / decoder.
    * A flow-matching Euler scheduler.

    Text encoding is assumed to have been performed offline (see
    ``preprocess/vectorize.py``), so the pipeline works with pre-computed
    ``prompt_embeds`` and ``pooled_prompt_embeds`` tensors rather than raw text.
    """

    model_cpu_offload_seq = "transformer->vae"

    def __init__(
        self,
        transformer: MaskDit_sd3_5,
        vae: AutoencoderKL,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        super().__init__()
        self.register_modules(
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    @torch.no_grad()
    def __call__(
        self,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        condition_dict: Optional[Dict[str, Any]] = None,
        num_inference_steps: int = 28,
        height: int = 512,
        width: int = 512,
        guidance_scale: float = 4.5,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        mu: Optional[float] = None,
        initial_latents: Optional[torch.Tensor] = None,
    ) -> Union[StableDiffusion3PipelineOutput, tuple]:
        """Generate images conditioned on pre-computed text embeddings and
        optional mask conditions.

        Args:
            prompt_embeds: Pre-computed prompt embeddings ``[B, seq_len, dim]``.
            pooled_prompt_embeds: Pre-computed pooled prompt embeddings ``[B, dim]``.
            condition_dict: Dictionary with ``"cond_types"`` and
                ``"cond_latents"`` lists.
            num_inference_steps: Number of denoising steps.
            height: Target image height.
            width: Target image width.
            guidance_scale: Classifier-free guidance scale.
            negative_prompt_embeds: Negative prompt embeddings for CFG.
            negative_pooled_prompt_embeds: Negative pooled prompt embeddings.
            generator: Random number generator(s).
            output_type: ``"pil"``, ``"pt"``, or ``"latent"``.
            return_dict: Whether to return a
                :class:`StableDiffusion3PipelineOutput`.
            mu: Dynamic shifting parameter.
            initial_latents: Optional starting latents.

        Returns:
            Generated images.
        """
        device = prompt_embeds.device
        batch_size = prompt_embeds.shape[0]
        latents_width = int(width / self.vae_scale_factor)
        latents_height = int(height / self.vae_scale_factor)
        num_channels_latents = self.transformer.config.in_channels

        if initial_latents is None:
            latents = _prepare_latents(
                batch_size,
                num_channels_latents,
                latents_height,
                latents_width,
                prompt_embeds.dtype,
                device,
                generator,
            )
        else:
            latents = initial_latents
        latent_dtype = latents.dtype

        # Prepare timesteps
        scheduler_kwargs = {}
        if self.scheduler.config.get("use_dynamic_shifting", None) and mu is None:
            _, _, h, w = latents.shape
            image_seq_len = (h // self.transformer.config.patch_size) * (
                w // self.transformer.config.patch_size
            )
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.16),
            )
            scheduler_kwargs["mu"] = mu
        elif mu is not None:
            scheduler_kwargs["mu"] = mu

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, **scheduler_kwargs
        )
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )

        use_cfg = negative_prompt_embeds is not None
        if use_cfg:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
            )

        for i, t in enumerate(timesteps):
            latent_model_input = (
                torch.cat([latents] * 2) if use_cfg else latents
            )
            timestep = t.expand(latent_model_input.shape[0])

            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                joint_attention_kwargs={},
                return_dict=False,
                condition_dict=condition_dict,
            )[0]

            if use_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            latents = self.scheduler.step(
                noise_pred, t, latents, return_dict=False
            )[0]

            if latents.dtype != latent_dtype:
                latents = latents.to(dtype=latent_dtype)

        if output_type == "latent":
            image = latents
        else:
            latents = (
                latents / self.vae.config.scaling_factor
            ) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        if not return_dict:
            return (image,)
        return StableDiffusion3PipelineOutput(images=image)
