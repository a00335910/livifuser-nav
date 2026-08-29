"""Measured LDS-03 nominal observation model for simulation.

The analytic ray caster remains deterministic ground truth.  This module is the
separate, downstream policy-observation layer required by preregistration
section 13.1: it resamples the empirical beam-count histogram, derives that
scan's angular increment, adds measured repeatability noise before 1 mm
quantization, and represents both genuine and stochastic no-returns using the
physical driver's zero/NaN mixture.

Only the stable-surface stochastic miss probability is injected.  The measured
3.34% aggregate no-return occupancy is deliberately retained in the versioned
configuration as an excluded quantity; it is mostly open-space geometry that
the ray caster already produces and adding it again would corrupt C0.

Stdlib only and Python 3.10 compatible so the same implementation runs on the
ROS host and the Windows analysis host.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

from .analytic_lidar import (
    AnalyticLidarGeometry,
    LaserSpecification,
    Pose2D,
    simulate_ranges,
)

SUPPORTED_SCHEMA_VERSION = "1.0.0"
EXPECTED_INCREMENT_RULE = "2*pi/(beam_count+1)"


@dataclass(frozen=True)
class LidarCondition:
    """Frozen policy-visible LiDAR condition applied after ideal ray casting."""

    name: str
    range_noise_sigma_m: float
    missing_return_probability: float
    structured_dropout_width_rad: float = 0.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("LiDAR condition name must not be empty")
        if not math.isfinite(self.range_noise_sigma_m) or self.range_noise_sigma_m <= 0.0:
            raise ValueError("condition range-noise sigma must be positive")
        if not 0.0 <= self.missing_return_probability < 1.0:
            raise ValueError("condition missing-return probability must be in [0, 1)")
        if not 0.0 <= self.structured_dropout_width_rad < math.tau:
            raise ValueError("structured dropout width must be in [0, 2*pi)")


# §13.2 values are total observation levels, not extra noise added on top of C0.
LIDAR_CONDITIONS: dict[str, LidarCondition] = {
    "C3a": LidarCondition("C3a", 0.0154, 0.01),
    "C3b": LidarCondition("C3b", 0.0616, 0.05, math.radians(30.0)),
}


@dataclass(frozen=True)
class EmpiricalHistogram:
    """Integer-valued empirical distribution with exact source weights."""

    values: tuple[int, ...]
    weights: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.values or len(self.values) != len(self.weights):
            raise ValueError("histogram values and weights must be non-empty and aligned")
        if tuple(sorted(self.values)) != self.values or len(set(self.values)) != len(
            self.values
        ):
            raise ValueError("histogram values must be unique and sorted")
        if any(value <= 0 for value in self.values):
            raise ValueError("histogram values must be positive")
        if any(weight <= 0 for weight in self.weights):
            raise ValueError("histogram weights must be positive")

    @property
    def total_weight(self) -> int:
        return sum(self.weights)

    def sample(self, generator: random.Random) -> int:
        """Sample without floating-point probability normalization."""

        draw = generator.randrange(self.total_weight)
        cumulative = 0
        for value, weight in zip(self.values, self.weights, strict=True):
            cumulative += weight
            if draw < cumulative:
                return value
        raise AssertionError("histogram draw escaped its cumulative weight")


@dataclass(frozen=True)
class NoReturnEncoding:
    zero_count: int
    nan_count: int

    def __post_init__(self) -> None:
        if self.zero_count <= 0 or self.nan_count <= 0:
            raise ValueError("both no-return encodings require positive source counts")

    @property
    def total_count(self) -> int:
        return self.zero_count + self.nan_count

    @property
    def zero_probability(self) -> float:
        return self.zero_count / self.total_count

    def sample(self, generator: random.Random) -> float:
        draw = generator.randrange(self.total_count)
        return 0.0 if draw < self.zero_count else math.nan


@dataclass(frozen=True)
class Lds03ObservationModel:
    schema_version: str
    name: str
    source_artifact_path: str
    source_artifact_sha256: str
    source_scan_count: int
    beam_counts: EmpiricalHistogram
    angle_min_rad: float
    angle_max_rad: float
    scan_interval_sec: float
    range_noise_sigma_m: float
    range_quantization_step_m: float
    stochastic_missing_return_probability: float
    no_return_encoding: NoReturnEncoding
    excluded_aggregate_no_return_occupancy: float

    def __post_init__(self) -> None:
        if self.schema_version != SUPPORTED_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported observation-model schema: {self.schema_version}"
            )
        if self.source_scan_count != self.beam_counts.total_weight:
            raise ValueError("beam-count histogram does not sum to source scan count")
        if not self.source_artifact_sha256 or len(self.source_artifact_sha256) != 64:
            raise ValueError("source artifact SHA-256 must contain 64 hexadecimal digits")
        try:
            int(self.source_artifact_sha256, 16)
        except ValueError as error:
            raise ValueError("source artifact SHA-256 is not hexadecimal") from error
        finite_positive = (
            self.scan_interval_sec,
            self.range_noise_sigma_m,
            self.range_quantization_step_m,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in finite_positive):
            raise ValueError("scan interval, noise sigma, and quantization must be positive")
        if not 0.0 <= self.stochastic_missing_return_probability < 1.0:
            raise ValueError("stochastic missing-return probability must be in [0, 1)")
        if not 0.0 <= self.excluded_aggregate_no_return_occupancy < 1.0:
            raise ValueError("excluded aggregate no-return occupancy must be in [0, 1)")
        if not math.isclose(self.angle_min_rad, 0.0, abs_tol=1e-12):
            raise ValueError("measured LDS-03 angle_min must be zero")
        if not math.isclose(self.angle_max_rad, math.tau, abs_tol=1e-12):
            raise ValueError("measured LDS-03 angle_max must be 2*pi")

    def specification_for(
        self, geometry: AnalyticLidarGeometry, beam_count: int
    ) -> LaserSpecification:
        if beam_count not in self.beam_counts.values:
            raise ValueError(f"beam count {beam_count} is outside the measured support")
        return LaserSpecification(
            beam_count=beam_count,
            angle_min_rad=self.angle_min_rad,
            angle_max_rad=self.angle_max_rad,
            scan_time_sec=self.scan_interval_sec,
            range_min_m=geometry.laser.range_min_m,
            range_max_m=geometry.laser.range_max_m,
            frame_id=geometry.laser.frame_id,
        )


@dataclass(frozen=True)
class SimulatedLds03Scan:
    specification: LaserSpecification
    ranges: tuple[float, ...]
    geometric_no_return_count: int
    stochastic_missing_return_count: int
    structured_missing_return_count: int = 0
    condition: str = "C0"

    def __post_init__(self) -> None:
        if len(self.ranges) != self.specification.beam_count:
            raise ValueError("scan ranges do not match the sampled beam count")


def _finite_float(value: object, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def load_observation_model(path: Path) -> Lds03ObservationModel:
    """Load the tracked copy of the measured nominal observation contract."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    histogram_payload = payload["beam_count_histogram"]
    if not isinstance(histogram_payload, dict):
        raise ValueError("beam_count_histogram must be an object")
    histogram_items = sorted(
        ((int(value), int(weight)) for value, weight in histogram_payload.items()),
        key=lambda item: item[0],
    )
    encoding_payload = payload["no_return_encoding_counts"]
    angular_payload = payload["angular_geometry"]
    if angular_payload["increment_rule"] != EXPECTED_INCREMENT_RULE:
        raise ValueError("unsupported angular increment rule")
    excluded_payload = payload["excluded_from_nominal_dropout"]
    source_payload = payload["source_artifact"]
    return Lds03ObservationModel(
        schema_version=str(payload["schema_version"]),
        name=str(payload["name"]),
        source_artifact_path=str(source_payload["path"]),
        source_artifact_sha256=str(source_payload["sha256"]).upper(),
        source_scan_count=int(source_payload["scan_count"]),
        beam_counts=EmpiricalHistogram(
            tuple(value for value, _ in histogram_items),
            tuple(weight for _, weight in histogram_items),
        ),
        angle_min_rad=_finite_float(angular_payload["angle_min_rad"], "angle_min_rad"),
        angle_max_rad=_finite_float(angular_payload["angle_max_rad"], "angle_max_rad"),
        scan_interval_sec=_finite_float(payload["scan_interval_sec"], "scan_interval_sec"),
        range_noise_sigma_m=_finite_float(
            payload["range_noise_sigma_m"], "range_noise_sigma_m"
        ),
        range_quantization_step_m=_finite_float(
            payload["range_quantization_step_m"], "range_quantization_step_m"
        ),
        stochastic_missing_return_probability=_finite_float(
            payload["stochastic_missing_return_probability"],
            "stochastic_missing_return_probability",
        ),
        no_return_encoding=NoReturnEncoding(
            zero_count=int(encoding_payload["zero"]),
            nan_count=int(encoding_payload["nan"]),
        ),
        excluded_aggregate_no_return_occupancy=_finite_float(
            excluded_payload["aggregate_no_return_occupancy"],
            "excluded aggregate no-return occupancy",
        ),
    )


def _quantize_positive(value: float, step_m: float) -> float:
    """Round a positive range to the nearest lattice point, half upward."""

    ticks = math.floor(value / step_m + 0.5)
    return ticks * step_m


def condition_for(model: Lds03ObservationModel, name: str) -> LidarCondition:
    """Resolve one frozen condition, deriving C0 from the measured model."""

    if name == "C0":
        return LidarCondition(
            "C0",
            model.range_noise_sigma_m,
            model.stochastic_missing_return_probability,
        )
    try:
        return LIDAR_CONDITIONS[name]
    except KeyError as error:
        raise ValueError(f"unknown LiDAR condition: {name}") from error


def apply_observation_condition(
    ideal_ranges: tuple[float, ...],
    specification: LaserSpecification,
    model: Lds03ObservationModel,
    generator: random.Random,
    condition: LidarCondition,
) -> SimulatedLds03Scan:
    """Apply a frozen C0/C3 observation condition to ideal ray-cast ranges."""

    if len(ideal_ranges) != specification.beam_count:
        raise ValueError("ideal ranges do not match the sampled beam count")
    observed: list[float] = []
    geometric_no_returns = 0
    stochastic_misses = 0
    structured_misses = 0
    sector_start = (
        generator.random() * math.tau
        if condition.structured_dropout_width_rad > 0.0
        else None
    )
    for index, ideal in enumerate(ideal_ranges):
        bearing = specification.angle_min_rad + index * specification.angle_increment_rad
        if (
            sector_start is not None
            and (bearing - sector_start) % math.tau
            < condition.structured_dropout_width_rad
        ):
            structured_misses += 1
            observed.append(model.no_return_encoding.sample(generator))
            continue
        if not math.isfinite(ideal):
            geometric_no_returns += 1
            observed.append(model.no_return_encoding.sample(generator))
            continue

        noisy = ideal + generator.gauss(0.0, condition.range_noise_sigma_m)
        quantized = _quantize_positive(noisy, model.range_quantization_step_m)
        if not specification.range_min_m <= quantized <= specification.range_max_m:
            geometric_no_returns += 1
            observed.append(model.no_return_encoding.sample(generator))
            continue
        if generator.random() < condition.missing_return_probability:
            stochastic_misses += 1
            observed.append(model.no_return_encoding.sample(generator))
            continue
        observed.append(quantized)

    return SimulatedLds03Scan(
        specification=specification,
        ranges=tuple(observed),
        geometric_no_return_count=geometric_no_returns,
        stochastic_missing_return_count=stochastic_misses,
        structured_missing_return_count=structured_misses,
        condition=condition.name,
    )


def apply_nominal_observation(
    ideal_ranges: tuple[float, ...],
    specification: LaserSpecification,
    model: Lds03ObservationModel,
    generator: random.Random,
) -> SimulatedLds03Scan:
    """Apply the measured C0 observation model (compatibility entry point)."""

    return apply_observation_condition(
        ideal_ranges,
        specification,
        model,
        generator,
        condition_for(model, "C0"),
    )


def simulate_observation(
    geometry: AnalyticLidarGeometry,
    pose: Pose2D,
    model: Lds03ObservationModel,
    generator: random.Random,
    condition: str = "C0",
) -> SimulatedLds03Scan:
    """Ray-cast and apply one frozen C0/C3 observation condition."""

    beam_count = model.beam_counts.sample(generator)
    specification = model.specification_for(geometry, beam_count)
    ideal = simulate_ranges(geometry, pose, specification=specification)
    return apply_observation_condition(
        ideal,
        specification,
        model,
        generator,
        condition_for(model, condition),
    )
