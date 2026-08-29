"""Minimal locked-shape LiViFuser geometry fusion and ACT policy."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

#: Locked TurtleBot3 Burger command envelope: metres per second, radians per
#: second. Both the tanh output scaling and the loss normalization use it, so
#: the two channels contribute to the loss in proportion to their own range.
ACTION_SCALE = (0.10, 0.50)

#: Architecture v1.1 §8.1 offline-evaluable ablations. "full" is the locked
#: model; every other variant removes exactly one mechanism so a comparison
#: isolates it. Variants construct only the modules they use, so parameter
#: counts are honest, but the "full" construction order is unchanged and keeps
#: its seeded initialization bit-for-bit.
VARIANTS = (
    "full",  # locked v1.1 model
    "lidar_only",  # §8.1 LiDAR-only learned policy
    "rgb_only",  # §8.1 RGB-only DINO policy
    "concat",  # §8.1 RGB–LiDAR concatenation, no cross-attention
    "no_fov_mask",  # §8.1 cross-attention without the geometric FOV mask
    "no_gate",  # §8.1 cross-attention without gating
    "no_temporal",  # §8.1 full model without temporal memory
)


@dataclass(frozen=True, slots=True)
class ModelDimensions:
    width: int = 256
    heads: int = 4
    visual_input: int = 384
    visual_tokens: int = 49
    lidar_tokens: int = 80
    context_k: int = 8
    horizon_h: int = 8


class CircularConv1d(nn.Module):
    def __init__(self, inputs: int, outputs: int, kernel_size: int) -> None:
        super().__init__()
        self.padding = kernel_size // 2
        self.conv = nn.Conv1d(inputs, outputs, kernel_size, padding=0)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = F.pad(values, (self.padding, self.padding), mode="circular")
        return self.conv(values)


class GeometryCrossAttention(nn.Module):
    """Per-sample compatibility-masked multi-head cross-attention."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.width = width
        self.heads = heads
        self.head_width = width // heads
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.output = nn.Linear(width, width)

    def forward(
        self,
        lidar: torch.Tensor,
        visual: torch.Tensor,
        compatibility: torch.Tensor,
        in_fov: torch.Tensor,
    ) -> torch.Tensor:
        batch, lidar_count, _ = lidar.shape
        visual_count = visual.shape[1]
        query = self.query(lidar).view(batch, lidar_count, self.heads, self.head_width)
        key = self.key(visual).view(batch, visual_count, self.heads, self.head_width)
        value = self.value(visual).view(batch, visual_count, self.heads, self.head_width)
        logits = torch.einsum("blhd,bshd->bhls", query, key) / math.sqrt(self.head_width)
        safe_mask = compatibility.clone()
        safe_mask[~in_fov, 0] = True
        logits = logits.masked_fill(~safe_mask[:, None], torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)
        attended = torch.einsum("bhls,bshd->blhd", weights, value).reshape(
            batch, lidar_count, self.width
        )
        return self.output(attended) * in_fov.unsqueeze(-1)


class LiViFuserPolicy(nn.Module):
    """Frozen-feature policy matching Architecture v1.1's locked Stage 2 path."""

    def __init__(
        self, dimensions: ModelDimensions | None = None, variant: str = "full"
    ) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        self.variant = variant
        self.dimensions = dimensions or ModelDimensions()
        dims = self.dimensions
        uses_vision = variant != "lidar_only"
        uses_lidar = variant != "rgb_only"
        uses_cross = variant in ("full", "no_fov_mask", "no_gate", "no_temporal")
        if uses_vision:
            self.visual_projection = nn.Sequential(
                nn.LayerNorm(dims.visual_input), nn.Linear(dims.visual_input, dims.width)
            )
        if uses_lidar:
            self.lidar_encoder = nn.Sequential(
                CircularConv1d(4, 128, 5),
                nn.GELU(),
                CircularConv1d(128, dims.width, 3),
                nn.GELU(),
            )
            self.lidar_norm = nn.LayerNorm(dims.width)
        if uses_cross:
            self.cross_attention = GeometryCrossAttention(dims.width, dims.heads)
        if variant in ("full", "no_fov_mask", "no_temporal"):
            self.gate = nn.Sequential(nn.Linear(dims.width * 2, dims.width), nn.Sigmoid())
        if variant in ("full", "no_gate", "no_temporal", "lidar_only"):
            self.rear_residual = nn.Sequential(
                nn.Linear(dims.width, dims.width), nn.GELU(), nn.Linear(dims.width, dims.width)
            )
        if variant == "concat":
            self.concat_projection = nn.Sequential(
                nn.Linear(dims.width * 2, dims.width),
                nn.GELU(),
                nn.Linear(dims.width, dims.width),
            )
        self.fusion_norm = nn.LayerNorm(dims.width)
        self.goal_embedding = nn.Sequential(nn.Linear(3, 32), nn.GELU(), nn.Linear(32, 32))
        self.state_embedding = nn.Sequential(nn.Linear(2, 32), nn.GELU(), nn.Linear(32, 32))
        if variant == "no_temporal":
            # Same input contract as the GRU, applied to the current step only,
            # so the comparison isolates memory rather than input access.
            self.temporal = nn.Sequential(
                nn.Linear(dims.width + 64, dims.width),
                nn.GELU(),
                nn.Linear(dims.width, dims.width),
            )
        else:
            self.temporal = nn.GRU(dims.width + 64, dims.width, batch_first=True)
        self.horizon_queries = nn.Parameter(torch.empty(dims.horizon_h, dims.width))
        nn.init.normal_(self.horizon_queries, std=0.02)
        self.action_decoder = nn.MultiheadAttention(
            dims.width, dims.heads, batch_first=True
        )
        self.action_head = nn.Sequential(
            nn.LayerNorm(dims.width),
            nn.Linear(dims.width, dims.width),
            nn.GELU(),
            nn.Linear(dims.width, 4),
        )
        self.register_buffer("action_scale", torch.tensor(ACTION_SCALE))

    def forward(
        self,
        visual_tokens: torch.Tensor,
        lidar_features: torch.Tensor,
        visual_mask: torch.Tensor,
        in_fov: torch.Tensor,
        goal: torch.Tensor,
        robot_state: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        dims = self.dimensions
        variant = self.variant
        batch, context = goal.shape[:2]
        if context != dims.context_k:
            raise ValueError(f"expected context K={dims.context_k}, got {context}")
        flat = batch * context
        if variant != "lidar_only":
            visual = self.visual_projection(
                visual_tokens.reshape(flat, dims.visual_tokens, dims.visual_input)
            )
        if variant != "rgb_only":
            lidar_input = lidar_features.reshape(flat, dims.lidar_tokens, 4).transpose(1, 2)
            lidar = self.lidar_norm(self.lidar_encoder(lidar_input).transpose(1, 2))
        gate = None
        if variant in ("full", "no_temporal"):
            flat_mask = visual_mask.reshape(flat, dims.lidar_tokens, dims.visual_tokens)
            flat_fov = in_fov.reshape(flat, dims.lidar_tokens)
            cross = self.cross_attention(lidar, visual, flat_mask, flat_fov)
            gate = self.gate(torch.cat((cross, lidar), dim=-1))
            front = gate * cross + (1.0 - gate) * lidar
            rear = lidar + self.rear_residual(lidar)
            fused_flat = torch.where(flat_fov.unsqueeze(-1), front, rear)
        elif variant == "no_fov_mask":
            open_mask = visual_mask.new_ones(
                (flat, dims.lidar_tokens, dims.visual_tokens), dtype=torch.bool
            )
            open_fov = in_fov.new_ones((flat, dims.lidar_tokens), dtype=torch.bool)
            cross = self.cross_attention(lidar, visual, open_mask, open_fov)
            gate = self.gate(torch.cat((cross, lidar), dim=-1))
            fused_flat = gate * cross + (1.0 - gate) * lidar
        elif variant == "no_gate":
            flat_mask = visual_mask.reshape(flat, dims.lidar_tokens, dims.visual_tokens)
            flat_fov = in_fov.reshape(flat, dims.lidar_tokens)
            cross = self.cross_attention(lidar, visual, flat_mask, flat_fov)
            front = cross + lidar
            rear = lidar + self.rear_residual(lidar)
            fused_flat = torch.where(flat_fov.unsqueeze(-1), front, rear)
        elif variant == "concat":
            pooled_visual = visual.mean(dim=1, keepdim=True).expand(
                -1, dims.lidar_tokens, -1
            )
            fused_flat = self.concat_projection(torch.cat((lidar, pooled_visual), dim=-1))
        elif variant == "lidar_only":
            fused_flat = lidar + self.rear_residual(lidar)
        else:  # rgb_only
            fused_flat = visual
        token_count = dims.visual_tokens if variant == "rgb_only" else dims.lidar_tokens
        fused = self.fusion_norm(fused_flat).reshape(batch, context, token_count, dims.width)
        pooled = fused.mean(dim=2)
        temporal_input = torch.cat(
            (pooled, self.goal_embedding(goal), self.state_embedding(robot_state)), dim=-1
        )
        if variant == "no_temporal":
            hidden = self.temporal(temporal_input[:, -1])
        else:
            _, hidden = self.temporal(temporal_input)
            hidden = hidden[-1]
        current_tokens = fused[:, -1]
        memory = torch.cat((current_tokens, hidden[:, None]), dim=1)
        queries = self.horizon_queries[None].expand(batch, -1, -1) + hidden[:, None]
        decoded, _ = self.action_decoder(queries, memory, memory, need_weights=False)
        raw = self.action_head(decoded)
        mean = torch.tanh(raw[..., :2]) * self.action_scale
        log_variance = raw[..., 2:]
        outputs = {
            "mean": mean,
            "log_variance": log_variance,
            "fused_tokens": fused,
            "hidden": hidden,
        }
        if gate is not None:
            outputs["gate"] = gate.reshape(batch, context, dims.lidar_tokens, dims.width)
        return outputs


def mean_warmup_loss(outputs: dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    scale = target.new_tensor(ACTION_SCALE)
    return F.mse_loss(outputs["mean"] / scale, target / scale)


def heteroscedastic_nll(
    outputs: dict[str, torch.Tensor], target: torch.Tensor
) -> torch.Tensor:
    scaled_error = (target - outputs["mean"]) / target.new_tensor(ACTION_SCALE)
    log_variance = outputs["log_variance"].clamp(-5.0, 2.0)
    return 0.5 * (torch.exp(-log_variance) * scaled_error.square() + log_variance).mean()

