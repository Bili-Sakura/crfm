#!/usr/bin/env python3
"""Convert an original CRFM MaskDit_sd3_5 checkpoint into a diffusers-compatible
pipeline directory that can be loaded with ``CRFMPipeline.from_pretrained()``.

Usage
-----
::

    python scripts/convert_checkpoint.py \
        --pretrained_model_name_or_path sd3.5_medium \
        --mmdit_ckpt path/to/model.safetensors \
        --output_dir converted_crfm_pipeline

The resulting ``output_dir`` can be loaded as::

    from src.pipeline_crfm import CRFMPipeline
    pipe = CRFMPipeline.from_pretrained("converted_crfm_pipeline")

If no ``--mmdit_ckpt`` is given the script still creates the pipeline directory
using only the base SD-3.5 weights (useful for testing the conversion flow).
"""

import argparse
import json
import os
import shutil

import torch
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel
from safetensors.torch import load_file, save_file

from src.models.sd3_mmdit import MaskDit_sd3_5


def convert_checkpoint(
    pretrained_model_name_or_path: str,
    output_dir: str,
    mmdit_ckpt: str | None = None,
    dtype_str: str = "fp32",
) -> str:
    """Build the full MaskDit_sd3_5 model, optionally load trained weights,
    and save everything in a diffusers pipeline layout.

    Args:
        pretrained_model_name_or_path: Path to the base SD-3.5 model (or
            HuggingFace Hub id).
        output_dir: Where to write the converted pipeline.
        mmdit_ckpt: Optional ``.safetensors`` file with trained parameters
            (as saved by the training hook).
        dtype_str: One of ``"fp32"``, ``"fp16"``, ``"bf16"``.

    Returns:
        The *output_dir* path.
    """
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    weight_dtype = dtype_map.get(dtype_str, torch.float32)

    # --- 1. Build the MaskDit_sd3_5 transformer ---
    print("[1/4] Loading base SD3 transformer …")
    sd3_transformer = SD3Transformer2DModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="transformer"
    )

    print("[2/4] Building MaskDit_sd3_5 …")
    transformer = MaskDit_sd3_5(sd3_transformer=sd3_transformer)

    if mmdit_ckpt is not None:
        print(f"  Loading trained weights from {mmdit_ckpt}")
        trained_sd = load_file(mmdit_ckpt)
        transformer.load_state_dict(trained_sd, strict=False)

    transformer = transformer.to(dtype=weight_dtype)

    # --- 2. Save transformer weights ---
    os.makedirs(output_dir, exist_ok=True)
    transformer_dir = os.path.join(output_dir, "transformer")
    os.makedirs(transformer_dir, exist_ok=True)

    state_dict = transformer.state_dict()
    save_file(state_dict, os.path.join(transformer_dir, "diffusion_pytorch_model.safetensors"))

    # Save a config so the model can be reconstructed later.
    sd3_config = dict(sd3_transformer.config) if hasattr(sd3_transformer, 'config') else {}
    transformer_config = {
        "_class_name": "MaskDit_sd3_5",
        "sd3_config": sd3_config,
    }
    with open(os.path.join(transformer_dir, "config.json"), "w") as f:
        json.dump(transformer_config, f, indent=2, default=str)

    # --- 3. Copy / link VAE & scheduler from the base model ---
    print("[3/4] Copying VAE and scheduler …")
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path, subfolder="vae"
    )
    vae.save_pretrained(os.path.join(output_dir, "vae"))

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        pretrained_model_name_or_path, subfolder="scheduler"
    )
    scheduler.save_pretrained(os.path.join(output_dir, "scheduler"))

    # --- 4. Write the top-level model_index.json ---
    print("[4/4] Writing model_index.json …")
    model_index = {
        "_class_name": "CRFMPipeline",
        "_diffusers_version": "0.30.0",
        "transformer": ["src.models.sd3_mmdit", "MaskDit_sd3_5"],
        "vae": ["diffusers", "AutoencoderKL"],
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    }
    with open(os.path.join(output_dir, "model_index.json"), "w") as f:
        json.dump(model_index, f, indent=2)

    print(f"✓ Converted pipeline saved to {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Convert a CRFM checkpoint to a diffusers pipeline."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to the base SD-3.5 model or HuggingFace Hub id.",
    )
    parser.add_argument(
        "--mmdit_ckpt",
        type=str,
        default=None,
        help="Path to the trained MaskDit .safetensors checkpoint.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="converted_crfm_pipeline",
        help="Directory to write the converted pipeline.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "bf16"],
        help="Data type for saved weights.",
    )
    args = parser.parse_args()
    convert_checkpoint(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        output_dir=args.output_dir,
        mmdit_ckpt=args.mmdit_ckpt,
        dtype_str=args.dtype,
    )


if __name__ == "__main__":
    main()
