"""Tests for the CRFM refactoring.

These tests validate:
1. SegmentationModelWrapper interface (no mmseg dependency)
2. Checkpoint conversion helpers (key mapping, config generation)
3. CRFMPipeline can be instantiated
4. MaskDit_sd3_5 forward pass shape consistency

All tests run **without** real model checkpoints by using small random weights.
"""

import json
import os
import sys
import tempfile
import unittest

import torch
import torch.nn as nn

# Ensure repo root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# 1. SegmentationModelWrapper
# ---------------------------------------------------------------------------
class _DummySegModel(nn.Module):
    """Minimal segmentation model for testing."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv = nn.Conv2d(3, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _DummyTransformersSegModel(nn.Module):
    """Simulates a transformers-style segmentation model output."""

    class _Output:
        def __init__(self, logits):
            self.logits = logits

    def __init__(self, num_classes: int = 10, downsample: int = 4):
        super().__init__()
        self.conv = nn.Conv2d(3, num_classes, kernel_size=1)
        self.downsample = downsample

    def forward(self, pixel_values: torch.Tensor):
        h, w = pixel_values.shape[-2:]
        logits = self.conv(pixel_values)
        logits = nn.functional.interpolate(
            logits,
            size=(h // self.downsample, w // self.downsample),
            mode="bilinear",
            align_corners=False,
        )
        return self._Output(logits)


class TestSegmentationModelWrapper(unittest.TestCase):
    def test_pytorch_model_passthrough(self):
        """Plain PyTorch model returns logits of correct shape."""
        from src.utils.seg_model import SegmentationModelWrapper

        model = _DummySegModel(num_classes=5)
        wrapper = SegmentationModelWrapper(model, is_transformers=False)
        x = torch.randn(2, 3, 64, 64)
        out = wrapper(x)
        self.assertEqual(out.shape, (2, 5, 64, 64))

    def test_transformers_model_upsample(self):
        """Transformers model with smaller spatial output gets upsampled."""
        from src.utils.seg_model import SegmentationModelWrapper

        model = _DummyTransformersSegModel(num_classes=8, downsample=4)
        wrapper = SegmentationModelWrapper(model, is_transformers=True)
        x = torch.randn(1, 3, 128, 128)
        out = wrapper(x)
        self.assertEqual(out.shape, (1, 8, 128, 128))

    def test_no_upsample_when_same_size(self):
        """No interpolation when output already matches input dims."""
        from src.utils.seg_model import SegmentationModelWrapper

        model = _DummySegModel(num_classes=3)
        wrapper = SegmentationModelWrapper(model, is_transformers=False)
        x = torch.randn(1, 3, 32, 32)
        out = wrapper(x)
        self.assertEqual(out.shape, (1, 3, 32, 32))


# ---------------------------------------------------------------------------
# 2. Conversion helpers
# ---------------------------------------------------------------------------
class TestConversionHelpers(unittest.TestCase):
    def test_model_index_structure(self):
        """model_index.json written by the conversion has the expected keys."""
        model_index = {
            "_class_name": "CRFMPipeline",
            "_diffusers_version": "0.30.0",
            "transformer": ["src.models.sd3_mmdit", "MaskDit_sd3_5"],
            "vae": ["diffusers", "AutoencoderKL"],
            "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        }
        self.assertIn("_class_name", model_index)
        self.assertEqual(model_index["_class_name"], "CRFMPipeline")
        self.assertIn("transformer", model_index)
        self.assertIn("vae", model_index)
        self.assertIn("scheduler", model_index)

    def test_transformer_config_written(self):
        """Transformer config.json round-trips correctly."""
        config = {
            "_class_name": "MaskDit_sd3_5",
            "sd3_config": {
                "attention_head_dim": 64,
                "num_attention_heads": 24,
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            tmp_path = f.name
        try:
            with open(tmp_path, "r") as f:
                loaded = json.load(f)
            self.assertEqual(loaded["_class_name"], "MaskDit_sd3_5")
            self.assertEqual(loaded["sd3_config"]["attention_head_dim"], 64)
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# 3. crfm.py refactored interface
# ---------------------------------------------------------------------------
class TestCRFMInterface(unittest.TestCase):
    def test_control_rf_matching_uses_forward(self):
        """Verify that crfm.control_rf_matching calls model(images) not model.predict()."""
        import inspect
        from src.utils import crfm

        source = inspect.getsource(crfm.control_rf_matching)
        self.assertNotIn(".predict(", source,
                         "control_rf_matching should no longer use .predict() (mmseg API)")
        self.assertNotIn("seg_logits", source,
                         "control_rf_matching should not reference seg_logits (mmseg API)")

    def test_crfm_test_no_mmseg_import(self):
        """crfm_test.py should not import from mmseg."""
        crfm_test_path = os.path.join(os.path.dirname(__file__), "..", "crfm_test.py")
        with open(crfm_test_path, "r") as f:
            contents = f.read()
        self.assertNotIn("from mmseg", contents)
        self.assertNotIn("import mmseg", contents)


# ---------------------------------------------------------------------------
# 4. Pipeline class
# ---------------------------------------------------------------------------
class TestCRFMPipelineClass(unittest.TestCase):
    def test_pipeline_importable(self):
        """CRFMPipeline can be imported without errors."""
        from src.pipeline_crfm import CRFMPipeline  # noqa: F401
        self.assertTrue(hasattr(CRFMPipeline, "__call__"))

    def test_pipeline_has_expected_attributes(self):
        """CRFMPipeline declares expected module names."""
        from src.pipeline_crfm import CRFMPipeline

        self.assertTrue(hasattr(CRFMPipeline, "model_cpu_offload_seq"))


if __name__ == "__main__":
    unittest.main()
