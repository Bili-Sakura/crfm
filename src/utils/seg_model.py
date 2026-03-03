"""
Segmentation model utilities that replace mmseg dependency.

Provides a generic wrapper for segmentation models (PyTorch or transformers-based)
to be used as the conditional model in the CRFM sampling process.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationModelWrapper(nn.Module):
    """Wraps a segmentation model to provide a standard interface.

    The wrapped model accepts image tensors of shape [B, 3, H, W] (normalized
    with ImageNet mean/std) and returns logits of shape [B, num_classes, H, W].
    """

    def __init__(self, model: nn.Module, is_transformers: bool = False):
        super().__init__()
        self.model = model
        self.is_transformers = is_transformers

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.is_transformers:
            output = self.model(pixel_values=pixel_values)
            logits = output.logits
        else:
            logits = self.model(pixel_values)
            if not isinstance(logits, torch.Tensor):
                logits = logits[0] if isinstance(logits, (tuple, list)) else logits

        # Ensure output spatial dims match input
        if logits.shape[-2:] != pixel_values.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=pixel_values.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return logits


def load_segmentation_model(
    model_path: str,
    checkpoint_path: Optional[str] = None,
    num_classes: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> SegmentationModelWrapper:
    """Load a segmentation model from a transformers pretrained path or a
    PyTorch checkpoint.

    Args:
        model_path: Path to a transformers pretrained model directory, or a
            HuggingFace Hub model id (e.g. ``"nvidia/segformer-b0-finetuned-ade-512-512"``).
        checkpoint_path: Optional path to a PyTorch ``.pth`` checkpoint.  When
            provided together with *model_path* pointing to a transformers model
            the checkpoint is **ignored** (the transformers weights take precedence).
        num_classes: Number of segmentation classes (only used when loading a
            plain PyTorch checkpoint without a transformers config).
        device: Target device.
        dtype: Target dtype.

    Returns:
        A :class:`SegmentationModelWrapper` ready for evaluation.
    """

    try:
        from transformers import AutoModelForSemanticSegmentation

        model = AutoModelForSemanticSegmentation.from_pretrained(model_path)
        model = model.to(device=device, dtype=dtype)
        return SegmentationModelWrapper(model, is_transformers=True)
    except Exception:
        pass

    if checkpoint_path is not None:
        state_dict = torch.load(
            checkpoint_path, map_location=device, weights_only=True
        )
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        from torchvision.models.segmentation import deeplabv3_resnet50

        model = deeplabv3_resnet50(num_classes=num_classes or 21)
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device=device, dtype=dtype)
        return SegmentationModelWrapper(model, is_transformers=False)

    raise ValueError(
        f"Could not load segmentation model from model_path={model_path!r}, "
        f"checkpoint_path={checkpoint_path!r}. Provide either a valid "
        "transformers model path or a PyTorch checkpoint."
    )
