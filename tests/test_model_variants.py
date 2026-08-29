"""Unit tests for the §8.1 ablation variants of the fusion policy.

Each variant must remove exactly the mechanism it names — proven here by
input-invariance (an ablated input cannot change the output) and by module
presence — while leaving the locked "full" construction untouched.

Skipped where PyTorch is unavailable (the ROS host).
"""

from __future__ import annotations

import unittest

try:  # pragma: no cover - availability is what is being guarded
    import torch

    from livifuser_nav.model import (
        VARIANTS,
        LiViFuserPolicy,
        heteroscedastic_nll,
        mean_warmup_loss,
    )
    from tests.test_model import make_batch, make_target

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False

#: Parameter count recorded by the passed Stage 2 overfit gate. The default
#: construction must never drift from it silently.
LOCKED_FULL_PARAMETER_COUNT = 1_506_468


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is unavailable on this host")
class VariantConstructionTests(unittest.TestCase):
    def test_default_variant_is_the_locked_full_model(self) -> None:
        model = LiViFuserPolicy()
        self.assertEqual(model.variant, "full")
        self.assertEqual(
            sum(item.numel() for item in model.parameters()),
            LOCKED_FULL_PARAMETER_COUNT,
        )

    def test_unknown_variant_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown variant"):
            LiViFuserPolicy(variant="ensemble")

    def test_variants_construct_only_the_modules_they_use(self) -> None:
        expectations = {
            "lidar_only": {"visual_projection": False, "cross_attention": False},
            "rgb_only": {"lidar_encoder": False, "cross_attention": False},
            "concat": {"cross_attention": False, "concat_projection": True},
            "no_fov_mask": {"cross_attention": True, "rear_residual": False},
            "no_gate": {"gate": False, "rear_residual": True},
            "no_temporal": {"cross_attention": True, "gate": True},
            "full": {"cross_attention": True, "gate": True, "rear_residual": True},
        }
        for variant, modules in expectations.items():
            model = LiViFuserPolicy(variant=variant)
            for name, expected in modules.items():
                with self.subTest(variant=variant, module=name):
                    self.assertEqual(hasattr(model, name), expected)

    def test_no_temporal_replaces_the_gru(self) -> None:
        self.assertNotIsInstance(
            LiViFuserPolicy(variant="no_temporal").temporal, torch.nn.GRU
        )
        self.assertIsInstance(LiViFuserPolicy(variant="full").temporal, torch.nn.GRU)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is unavailable on this host")
class VariantForwardTests(unittest.TestCase):
    def forward(self, variant: str, batch: dict) -> dict:
        torch.manual_seed(11)
        model = LiViFuserPolicy(variant=variant)
        model.eval()
        with torch.no_grad():
            return model(**batch)

    def test_every_variant_produces_locked_output_shapes(self) -> None:
        batch = make_batch()
        expected = tuple(batch["goal"].shape[:1]) + (8, 2)
        for variant in VARIANTS:
            outputs = self.forward(variant, batch)
            with self.subTest(variant=variant):
                self.assertEqual(tuple(outputs["mean"].shape), expected)
                self.assertEqual(tuple(outputs["log_variance"].shape), expected)
                self.assertTrue(bool(torch.isfinite(outputs["mean"]).all()))

    def test_gate_output_exists_exactly_where_a_gate_exists(self) -> None:
        batch = make_batch()
        for variant in VARIANTS:
            outputs = self.forward(variant, batch)
            with self.subTest(variant=variant):
                self.assertEqual(
                    "gate" in outputs,
                    variant in ("full", "no_fov_mask", "no_temporal"),
                )

    def assert_invariant_to(self, variant: str, key: str) -> None:
        batch = make_batch()
        first = self.forward(variant, batch)["mean"]
        changed = dict(batch)
        generator = torch.Generator().manual_seed(99)
        changed[key] = torch.randn(
            batch[key].shape, generator=generator, dtype=torch.float32
        )
        second = self.forward(variant, changed)["mean"]
        torch.testing.assert_close(first, second)

    def test_lidar_only_ignores_visual_tokens(self) -> None:
        self.assert_invariant_to("lidar_only", "visual_tokens")

    def test_rgb_only_ignores_lidar_features(self) -> None:
        self.assert_invariant_to("rgb_only", "lidar_features")

    def test_no_fov_mask_ignores_the_geometry_mask(self) -> None:
        batch = make_batch()
        first = self.forward("no_fov_mask", batch)["mean"]
        changed = dict(batch)
        changed["visual_mask"] = torch.zeros_like(batch["visual_mask"])
        changed["in_fov"] = torch.zeros_like(batch["in_fov"])
        second = self.forward("no_fov_mask", changed)["mean"]
        torch.testing.assert_close(first, second)

    def test_full_model_uses_the_geometry_mask(self) -> None:
        batch = make_batch()
        first = self.forward("full", batch)["mean"]
        changed = dict(batch)
        changed["visual_mask"] = torch.zeros_like(batch["visual_mask"])
        changed["in_fov"] = torch.zeros_like(batch["in_fov"])
        second = self.forward("full", changed)["mean"]
        self.assertFalse(bool(torch.isclose(first, second).all()))

    def test_concat_uses_visual_information(self) -> None:
        batch = make_batch()
        first = self.forward("concat", batch)["mean"]
        changed = dict(batch)
        generator = torch.Generator().manual_seed(99)
        changed["visual_tokens"] = torch.randn(
            batch["visual_tokens"].shape, generator=generator
        )
        second = self.forward("concat", changed)["mean"]
        self.assertFalse(bool(torch.isclose(first, second).all()))

    def test_every_variant_trains_every_parameter(self) -> None:
        target = make_target()
        for variant in VARIANTS:
            torch.manual_seed(11)
            model = LiViFuserPolicy(variant=variant)
            batch = make_batch()
            outputs = model(**batch)
            loss = mean_warmup_loss(outputs, target) + heteroscedastic_nll(
                outputs, target
            )
            loss.backward()
            for name, parameter in model.named_parameters():
                with self.subTest(variant=variant, parameter=name):
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(bool(torch.isfinite(parameter.grad).all()))


if __name__ == "__main__":
    unittest.main()
