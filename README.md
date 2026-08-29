# LiViFuser-Nav

Geometry-aware fusion of dense visual features and sparse 2D LiDAR for local
navigation on a TurtleBot3 Burger, evaluated under preregistration.

The study is finished and the headline result is negative. A geometry-aware
fusion policy over frozen DINOv3 features did not improve on a LiDAR-only
baseline in closed loop. On the matched reference condition it reached the goal
in 9 of 60 episodes against 29 of 60 for LiDAR-only, and it collided more often.
The preregistered support criterion fails on all three of its required
comparisons.

This repository holds the system, the frozen protocol, and the evidence needed
to check that claim.

## What you may find useful here

**Measured LDS-03 behaviour that contradicts the datasheet.** If you are working
with a TurtleBot3 Burger, this is probably the most reusable part of the
repository. From 20 recordings and 15,295 scans:

- Measured scan rate is about 10.06 Hz, which matches the 10 Hz ROBOTIS
  documents. An earlier internal design assumption of 5 Hz was simply wrong;
  there are no duplicated scans at 10 Hz sampling.
- ROBOTIS gives angular resolution as 0.9 degrees, which implies 400 points per
  revolution, and notes that resolution varies because the 4 kHz sampling rate
  is fixed. In practice the variation happens *scan to scan within one run*:
  beam count moves between 396 and 404, and the driver rescales
  `angle_increment` to `2*pi/(beam_count+1)` each time. Beam index *i* therefore
  has no fixed bearing across scans. A precomputed bearing table, which is the
  obvious implementation, silently corrupts every LiDAR token, by up to 7.2
  degrees at the far beam on the longest run measured. Use the per-scan
  increment.
- The camera sits 8.4 cm below the scan plane, so a nearer return projects
  higher in the image, not lower.

Numbers and method are in
`artifacts/lidar/lds03_characterization_v1_2026-08-08.json`, produced by
`scripts/characterize_lds03.py`.

**A protocol frozen before the data existed.** The success criteria, corruption
conditions, episode schedule and analysis rules were fixed by cryptographic hash
before any confirmatory episode was generated. The schedule and the batch seal
are in `artifacts/`; the preregistration documents themselves are not published
here, and are available on request.

**An analysis written before the outcome was known.**
`src/livifuser_nav/closed_loop_analysis.py` and its tests were committed before
the batch finished, so the estimator, resampling scheme, replicate count, RNG
seed and claim criteria could not be tuned toward a result.

## Reproducing the closed-loop numbers

The per-episode outcome records are in the repository, so the analysis runs
without downloading anything.

```bash
uv sync --dev
uv run python scripts/report_closed_loop_results.py \
  --root artifacts/evidence_meta/evidence/confirmatory_closed_loop_v1
```

That prints the success and collision tables per arm, condition and seed with
exact denominators, the paired contrasts with their bootstrap intervals, and the
criterion verdict. It should reproduce 505 accepted episodes and a full minus
LiDAR-only contrast of -0.333, 95% CI [-0.500, -0.150].

The batch seal in `artifacts/closed_loop_batch_seal_v1.json` records the
classification and terminal hash of all 888 attempts. It computes no rate and no
contrast, because the analysis runs after it.

## Running the tests

```bash
uv sync --dev
uv run python -m unittest discover -s tests -t .
uv run ruff check .
```

906 tests, with 33 skipping where PyTorch or ROS 2 is absent. The ROS-side suite
runs inside WSL:

```bash
wsl.exe -d Ubuntu-TB3 -- bash -lc 'source /opt/ros/humble/setup.bash \
  && source /mnt/d/LiViFuser/ros2_ws/install/setup.bash \
  && cd /mnt/d/LiViFuser && python3 -m unittest discover -s tests -t .'
```

Two of those tests start real `/cmd_vel` publishers. Each pins its own ROS domain
and sets `ROS_LOCALHOST_ONLY=1`, so nothing reaches a robot. Do not weaken that
isolation to fix a discovery problem.

Build the ROS workspace:

```bash
wsl.exe -d Ubuntu-TB3 -- bash -lc 'cd /mnt/d/LiViFuser/ros2_ws \
  && source /opt/ros/humble/setup.bash && colcon build --symlink-install'
```

## Safety

`livifuser_command_watchdog` is the only component permitted to publish
`/cmd_vel`. It clamps commands, enforces a 250 ms freshness timeout, and forces
zero on missing, stale, invalid or conflicting input. Read
`ros2_ws/src/livifuser_command_watchdog/` before running anything that could
move a robot. `scripts/replay_pilot_bag.py` is audit-only unless you pass
`--play`, and it cannot construct a velocity publisher otherwise.

## Layout

```text
src/livifuser_nav/   Platform-neutral logic: contracts, association, sampling,
                     export schema, learning data, model, training, evaluation,
                     closed-loop analysis. No ROS imports.
ros2_ws/src/         ROS 2 packages: command watchdog, goal publisher, episode
                     manager, simulation supervisor, bringup, interfaces.
scripts/             ROS-facing glue, exporters, audit and packaging tools.
tests/               Unit tests. ROS and PyTorch tests guard their imports.
config/              Frozen configuration, hashed into every result.
artifacts/           Small evidence records only (see below).
```

Anything deciding whether data is valid lives in `src/`, where it is unit
tested. A boundary bug and a beam-count bug have both hidden in untested script
code before.

## What is not in this repository

Raw recordings are excluded by size, not by choice: roughly 55 GB of MCAP bags
from the closed-loop study and the physical robot, plus a 146 GB simulation
episode dataset, feature caches, and model checkpoints. `artifacts/` carries the
small records that the reported numbers actually resolve to, and everything
larger is identified by SHA-256 so a copy can be verified against what was used.

## Status and limitations

The generalization unit is the world and the held-out set contains two, so the
exact sign test cannot fall below p = 0.5 even when both worlds agree. Three
training seeds were used, and seed-to-seed variation exceeded every contrast
measured: LiDAR-only success ranges from 17/20 to 2/20 across seeds of one
variant. The study is simulation-only; the physical robot contributed the sensor
observation model, not the confirmatory data. One condition, C1, covers a single
world for three of four arms because of a subset-selection defect that is
documented rather than hidden.

Architecture v1.1 is locked. The backbone used throughout is an ordinary DINOv3
ViT-S/16 (21.6M parameters), a recorded deviation from the locked ViT-S+/16
(29M). It is not described as S+/16 anywhere it was not.

## License

No license has been chosen yet, so default copyright applies and no permission
to reuse is granted. Open an issue if you want to use any of it.
