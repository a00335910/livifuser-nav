"""Unit tests for the locked-shape fusion policy, its gradients, and its losses.

Skipped where PyTorch is unavailable, which is the case on the ROS host: the
model is trained and evaluated on the Windows side and nothing in `scripts/`
imports it.
"""

from __future__ import annotations

import unittest

try:  # pragma: no cover - availability is what is being guarded
    import torch

    from livifuser_nav.model import (
        CircularConv1d,
        GeometryCrossAttention,
        LiViFuserPolicy,
        ModelDimensions,
        heteroscedastic_nll,
        mean_warmup_loss,
    )

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False

BATCH = 3
DIMENSIONS = ModelDimensions() if TORCH_AVAILABLE else None


def make_batch(seed: int = 0, *, all_out_of_view: bool = False) -> dict:
    """A structurally valid batch with the same invariants `tokenize_lidar` gives."""

    generator = torch.Generator().manual_seed(seed)
    dims = DIMENSIONS
    in_fov = torch.zeros(BATCH, dims.context_k, dims.lidar_tokens, dtype=torch.bool)
    if not all_out_of_view:
        in_fov[:, :, 8:20] = True
    # `tokenize_lidar` gives an in-view LiDAR token a small patch neighbourhood
    # and an out-of-view token an all-false row; mirror both here.
    shape = (BATCH, dims.context_k, dims.lidar_tokens, dims.visual_tokens)
    visual_mask = torch.rand(shape, generator=generator) < 0.12
    patch = torch.randint(
        0, dims.visual_tokens, shape[:-1], generator=generator
    )
    visual_mask.scatter_(-1, patch.unsqueeze(-1), True)
    visual_mask &= in_fov.unsqueeze(-1)
    return {
        "visual_tokens": torch.randn(
            BATCH, dims.context_k, dims.visual_tokens, dims.visual_input, generator=generator
        ),
        "lidar_features": torch.rand(
            BATCH, dims.context_k, dims.lidar_tokens, 4, generator=generator
        ),
        "visual_mask": visual_mask,
        "in_fov": in_fov,
        "goal": torch.randn(BATCH, dims.context_k, 3, generator=generator),
        "robot_state": torch.randn(BATCH, dims.context_k, 2, generator=generator),
    }


def make_target(seed: int = 7) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    scale = torch.tensor([0.10, 0.50])
    return (torch.rand(BATCH, DIMENSIONS.horizon_h, 2, generator=generator) * 2 - 1) * scale


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is unavailable on this host")
class ForwardShapeTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model = LiViFuserPolicy()
        self.batch = make_batch()
        with torch.no_grad():
            self.outputs = self.model(**self.batch)

    def test_output_shapes_match_the_locked_dimensions(self) -> None:
        dims = DIMENSIONS
        self.assertEqual(tuple(self.outputs["mean"].shape), (BATCH, dims.horizon_h, 2))
        self.assertEqual(tuple(self.outputs["log_variance"].shape), (BATCH, dims.horizon_h, 2))
        self.assertEqual(
            tuple(self.outputs["fused_tokens"].shape),
            (BATCH, dims.context_k, dims.lidar_tokens, dims.width),
        )
        self.assertEqual(
            tuple(self.outputs["gate"].shape),
            (BATCH, dims.context_k, dims.lidar_tokens, dims.width),
        )
        self.assertEqual(tuple(self.outputs["hidden"].shape), (BATCH, dims.width))

    def test_all_outputs_are_finite(self) -> None:
        for name, value in self.outputs.items():
            with self.subTest(output=name):
                self.assertTrue(bool(torch.isfinite(value).all()))

    def test_predicted_mean_respects_the_action_scale(self) -> None:
        mean = self.outputs["mean"]
        self.assertLessEqual(float(mean[..., 0].abs().max()), 0.10)
        self.assertLessEqual(float(mean[..., 1].abs().max()), 0.50)

    def test_gate_is_a_unit_interval_convex_weight(self) -> None:
        gate = self.outputs["gate"]
        self.assertGreaterEqual(float(gate.min()), 0.0)
        self.assertLessEqual(float(gate.max()), 1.0)

    def test_wrong_context_length_is_refused(self) -> None:
        batch = {name: value[:, :4] for name, value in self.batch.items()}
        with self.assertRaisesRegex(ValueError, "expected context K=8"):
            self.model(**batch)

    def test_forward_is_deterministic_in_eval(self) -> None:
        self.model.eval()
        with torch.no_grad():
            first = self.model(**self.batch)["mean"]
            second = self.model(**self.batch)["mean"]
        torch.testing.assert_close(first, second, rtol=0, atol=0)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is unavailable on this host")
class GeometryMaskingTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(1)
        self.model = LiViFuserPolicy().eval()

    def test_out_of_view_tokens_ignore_the_camera_entirely(self) -> None:
        # The rear path is LiDAR-only by construction: replacing the visual
        # tokens must not move a single out-of-view fused token.
        batch = make_batch(seed=2)
        with torch.no_grad():
            first = self.model(**batch)["fused_tokens"]
            replaced = dict(batch)
            replaced["visual_tokens"] = torch.randn_like(batch["visual_tokens"]) * 5.0
            second = self.model(**replaced)["fused_tokens"]
        rear = ~batch["in_fov"]
        torch.testing.assert_close(first[rear], second[rear], rtol=0, atol=0)
        self.assertFalse(bool(torch.equal(first[batch["in_fov"]], second[batch["in_fov"]])))

    def test_a_fully_occluded_scan_still_produces_finite_actions(self) -> None:
        # Every LiDAR token behind the camera: the masked softmax must not
        # divide by an all-masked row.
        batch = make_batch(seed=3, all_out_of_view=True)
        with torch.no_grad():
            outputs = self.model(**batch)
        self.assertTrue(bool(torch.isfinite(outputs["mean"]).all()))
        self.assertTrue(bool(torch.isfinite(outputs["fused_tokens"]).all()))

    def test_cross_attention_only_reads_compatible_patches(self) -> None:
        torch.manual_seed(4)
        attention = GeometryCrossAttention(DIMENSIONS.width, DIMENSIONS.heads).eval()
        lidar = torch.randn(1, 4, DIMENSIONS.width)
        visual = torch.randn(1, DIMENSIONS.visual_tokens, DIMENSIONS.width)
        compatibility = torch.zeros(1, 4, DIMENSIONS.visual_tokens, dtype=torch.bool)
        compatibility[0, :, :3] = True
        in_fov = torch.ones(1, 4, dtype=torch.bool)
        with torch.no_grad():
            baseline = attention(lidar, visual, compatibility, in_fov)
            perturbed_visual = visual.clone()
            perturbed_visual[0, 10:] += 3.0  # outside every compatibility row
            perturbed = attention(lidar, perturbed_visual, compatibility, in_fov)
        torch.testing.assert_close(baseline, perturbed, rtol=0, atol=0)

    def test_cross_attention_output_is_zero_outside_the_field_of_view(self) -> None:
        torch.manual_seed(5)
        attention = GeometryCrossAttention(DIMENSIONS.width, DIMENSIONS.heads).eval()
        lidar = torch.randn(1, 4, DIMENSIONS.width)
        visual = torch.randn(1, DIMENSIONS.visual_tokens, DIMENSIONS.width)
        compatibility = torch.zeros(1, 4, DIMENSIONS.visual_tokens, dtype=torch.bool)
        compatibility[0, 0, :5] = True
        in_fov = torch.tensor([[True, False, False, False]])
        with torch.no_grad():
            output = attention(lidar, visual, compatibility, in_fov)
        torch.testing.assert_close(output[0, 1:], torch.zeros(3, DIMENSIONS.width))
        self.assertGreater(float(output[0, 0].abs().max()), 0.0)

    def test_a_single_compatible_patch_is_read_with_weight_one(self) -> None:
        torch.manual_seed(11)
        attention = GeometryCrossAttention(DIMENSIONS.width, DIMENSIONS.heads).eval()
        lidar = torch.randn(1, 1, DIMENSIONS.width)
        visual = torch.randn(1, DIMENSIONS.visual_tokens, DIMENSIONS.width)
        compatibility = torch.zeros(1, 1, DIMENSIONS.visual_tokens, dtype=torch.bool)
        compatibility[0, 0, 23] = True
        with torch.no_grad():
            output = attention(lidar, visual, compatibility, torch.ones(1, 1, dtype=torch.bool))
            expected = attention.output(attention.value(visual[:, 23:24]))
        torch.testing.assert_close(output, expected, rtol=1e-6, atol=1e-6)

    def test_the_caller_supplied_mask_is_not_mutated(self) -> None:
        attention = GeometryCrossAttention(DIMENSIONS.width, DIMENSIONS.heads).eval()
        compatibility = torch.zeros(1, 2, DIMENSIONS.visual_tokens, dtype=torch.bool)
        compatibility[0, 0, 0] = True
        original = compatibility.clone()
        with torch.no_grad():
            attention(
                torch.randn(1, 2, DIMENSIONS.width),
                torch.randn(1, DIMENSIONS.visual_tokens, DIMENSIONS.width),
                compatibility,
                torch.tensor([[True, False]]),
            )
        torch.testing.assert_close(compatibility, original, rtol=0, atol=0)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is unavailable on this host")
class CircularConvolutionTests(unittest.TestCase):
    def test_length_is_preserved(self) -> None:
        layer = CircularConv1d(4, 8, 5)
        self.assertEqual(tuple(layer(torch.randn(2, 4, 80)).shape), (2, 8, 80))

    def test_the_sector_ring_wraps(self) -> None:
        # Rotating the scan must rotate the encoding: sector 0 and sector 79 are
        # neighbours on the robot, and zero padding would break that.
        torch.manual_seed(6)
        layer = CircularConv1d(4, 8, 5).eval()
        values = torch.randn(1, 4, 80)
        with torch.no_grad():
            rolled_output = layer(torch.roll(values, 17, dims=-1))
            output_rolled = torch.roll(layer(values), 17, dims=-1)
        torch.testing.assert_close(rolled_output, output_rolled, rtol=1e-5, atol=1e-6)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is unavailable on this host")
class LossTests(unittest.TestCase):
    def test_warmup_loss_is_zero_on_an_exact_match(self) -> None:
        target = make_target()
        loss = mean_warmup_loss({"mean": target.clone()}, target)
        self.assertAlmostEqual(float(loss), 0.0, places=12)

    def test_warmup_loss_normalizes_both_channels_equally(self) -> None:
        # A 0.01 m/s linear error and a 0.05 rad/s angular error are the same
        # fraction of their action ranges and must cost the same.
        target = torch.zeros(1, DIMENSIONS.horizon_h, 2)
        linear = target.clone()
        linear[..., 0] = 0.01
        angular = target.clone()
        angular[..., 1] = 0.05
        self.assertAlmostEqual(
            float(mean_warmup_loss({"mean": linear}, target)),
            float(mean_warmup_loss({"mean": angular}, target)),
            places=8,
        )

    def test_warmup_loss_grows_with_error(self) -> None:
        target = make_target()
        close = mean_warmup_loss({"mean": target * 0.9}, target)
        far = mean_warmup_loss({"mean": target * 0.1}, target)
        self.assertLess(float(close), float(far))

    def test_nll_prefers_a_variance_that_matches_the_error(self) -> None:
        target = torch.zeros(2, DIMENSIONS.horizon_h, 2)
        mean = torch.full_like(target, 0.0)
        mean[..., 0] = 0.10  # a one-unit normalized error on the linear channel
        losses = {}
        for log_variance in (-2.0, 0.0, 2.0):
            losses[log_variance] = float(
                heteroscedastic_nll(
                    {"mean": mean, "log_variance": torch.full_like(target, log_variance)},
                    target,
                )
            )
        self.assertLess(losses[0.0], losses[-2.0])
        self.assertLess(losses[0.0], losses[2.0])

    def test_nll_clamps_the_log_variance(self) -> None:
        target = make_target()
        outputs = {"mean": target.clone(), "log_variance": torch.full_like(target, -50.0)}
        clamped = {"mean": target.clone(), "log_variance": torch.full_like(target, -5.0)}
        self.assertTrue(torch.isfinite(heteroscedastic_nll(outputs, target)))
        self.assertAlmostEqual(
            float(heteroscedastic_nll(outputs, target)),
            float(heteroscedastic_nll(clamped, target)),
            places=12,
        )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is unavailable on this host")
class GradientTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(8)
        self.model = LiViFuserPolicy()
        self.batch = make_batch(seed=9)
        self.target = make_target()

    def test_every_parameter_receives_a_finite_gradient(self) -> None:
        loss = mean_warmup_loss(self.model(**self.batch), self.target)
        loss.backward()
        for name, parameter in self.model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
                self.assertGreater(float(parameter.grad.abs().sum()), 0.0)

    def test_the_nll_phase_also_trains_the_whole_model(self) -> None:
        loss = heteroscedastic_nll(self.model(**self.batch), self.target)
        loss.backward()
        for name, parameter in self.model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(bool(torch.isfinite(parameter.grad).all()))

    def test_a_fully_out_of_view_batch_keeps_gradients_finite(self) -> None:
        batch = make_batch(seed=10, all_out_of_view=True)
        loss = mean_warmup_loss(self.model(**batch), self.target)
        loss.backward()
        for name, parameter in self.model.named_parameters():
            with self.subTest(parameter=name):
                self.assertTrue(bool(torch.isfinite(parameter.grad).all()))

    def test_a_few_steps_reduce_the_loss(self) -> None:
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-3)
        losses = []
        for _ in range(10):
            optimizer.zero_grad(set_to_none=True)
            loss = mean_warmup_loss(self.model(**self.batch), self.target)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        self.assertLess(losses[-1], losses[0])


if __name__ == "__main__":
    unittest.main()
