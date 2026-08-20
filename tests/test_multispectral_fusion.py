from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from custom_fusion_modules import assert_fusion_context_empty
from train_multispectral_fusion import build_model


def _tensor_sum(value) -> torch.Tensor:
    tensors: list[torch.Tensor] = []

    def visit(item) -> None:
        if torch.is_tensor(item):
            tensors.append(item.float().mean())
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)

    visit(value)
    if not tensors:
        raise AssertionError("Model output did not contain tensors")
    return torch.stack(tensors).sum()


class MultispectralFusionTests(unittest.TestCase):
    def test_fusion_forward_gradient_and_empty_context(self) -> None:
        cases = [
            ("f1_p4_concat", ("R", "G", "B", "NIR", "RedEdge")),
            ("f2_multiscale_gated", ("R", "G", "B", "NIR", "RedEdge")),
            (
                "f3_index_guided_gated",
                ("R", "G", "B", "NIR", "RedEdge", "NDVI", "NDRE"),
            ),
        ]
        for mode, channels in cases:
            with self.subTest(mode=mode):
                model = build_model("yolov8n-seg.pt", mode, channels, seed=0)
                model.eval()
                output = model(torch.rand(1, len(channels), 64, 64))
                loss = _tensor_sum(output)
                loss.backward()
                assert_fusion_context_empty(model)

                spectral_gradients = [
                    parameter.grad
                    for name, parameter in model.named_parameters()
                    if "spectral_encoder" in name and parameter.requires_grad
                ]
                fusion_gradients = [
                    parameter.grad
                    for name, parameter in model.named_parameters()
                    if ".fusion." in name and parameter.requires_grad
                ]
                self.assertTrue(
                    spectral_gradients
                    and any(grad is not None and torch.isfinite(grad).all() for grad in spectral_gradients)
                )
                self.assertTrue(
                    fusion_gradients
                    and any(grad is not None and torch.isfinite(grad).all() for grad in fusion_gradients)
                )

    def test_fusion_checkpoint_roundtrip(self) -> None:
        channels = ("R", "G", "B", "NIR", "RedEdge")
        model = build_model("yolov8n-seg.pt", "f2_multiscale_gated", channels, seed=0)
        model.eval()
        with torch.no_grad():
            before = _tensor_sum(model(torch.rand(1, len(channels), 64, 64))).item()
        assert_fusion_context_empty(model)

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "fusion_roundtrip.pt"
            torch.save({"model": model}, checkpoint)
            loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"].float().eval()
            with torch.no_grad():
                after = _tensor_sum(loaded(torch.rand(1, len(channels), 64, 64))).item()
        assert_fusion_context_empty(loaded)
        self.assertTrue(torch.isfinite(torch.tensor([before, after])).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
