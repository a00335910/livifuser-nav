#!/usr/bin/env python3
"""Build the deterministic Kaggle T4x2 held-out evaluation notebook."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPOSITORY_ROOT / "notebooks" / "kaggle_t4x2_sim_heldout_evaluation_v1.ipynb"
AMENDMENT_SHA256 = "2CD7ADE1AC43FBC74975D9987E6C6052F5146B9FAD4CB97B1D46BEC996F4EE55"
REPAIR_SHA256 = "EB19516B2D84D4830A7A34B7EDB56DFBACE7E8C8E17866AEB9605B9929AC9357"
CODE_MANIFEST_SHA256 = "9E0A6F5176F290F46AC732575459053F5A7E95A8E8A2F53E67F6281B03517F74"
RESULT_ARCHIVE_SHA256 = "F5B7D9EAB29DD20CE6710E4B803EAA331A5D7C2E741E9330995A1EAE615B9AC7"
SCORE_ARCHIVE_SHA256 = "07116A629E296929D69EDA41E44CB6067CB6C751C735B66FD0A1B736D240751B"
SCORE_MANIFEST_SHA256 = "BFD5A21F150DCFCF12CD988821DE6901A1558ACC0EA183D7F8223940C2C1A729"


def source(value: str) -> list[str]:
    return [line + chr(10) for line in value.strip().splitlines()]


def code_cell(value: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source(value),
    }


def markdown_cell(value: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source(value),
    }


def notebook() -> dict[str, object]:
    preflight = f"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

INPUT = Path('/kaggle/input')
WORK = Path('/kaggle/working')
AMENDMENT_SHA256 = '{AMENDMENT_SHA256}'
REPAIR_SHA256 = '{REPAIR_SHA256}'
CODE_MANIFEST_SHA256 = '{CODE_MANIFEST_SHA256}'

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest().upper()

code_candidates = []
for path in INPUT.rglob('cloud_bundle_manifest.json'):
    try:
        raw = path.read_bytes()
        manifest = json.loads(raw)
    except Exception:
        continue
    if hashlib.sha256(raw).hexdigest().upper() == CODE_MANIFEST_SHA256:
        code_candidates.append((path, raw, manifest))
assert len(code_candidates) == 1, (
    f'expected one expanded held-out code manifest, found {{len(code_candidates)}}'
)
code_manifest_path, code_manifest_raw, code_manifest = code_candidates[0]
assert hashlib.sha256(code_manifest_raw).hexdigest().upper() == CODE_MANIFEST_SHA256
assert code_manifest['frozen_amendment_sha256'] == AMENDMENT_SHA256
assert code_manifest['execution_repair_sha256'] == REPAIR_SHA256
assert code_manifest['heldout_feature_or_outcome_included'] is False
REPO = code_manifest_path.parent
sys.path[:0] = [str(REPO / 'src'), str(REPO / 'scripts')]
from livifuser_nav.cloud_bundle import verify_cloud_bundle
verification = verify_cloud_bundle(REPO)
assert verification['manifest_sha256'].upper() == CODE_MANIFEST_SHA256
config = REPO / 'config' / 'simulation_sweep_v1.json'
amendment = (
    REPO / 'docs' / 'experiments'
    / 'PREREGISTRATION_SIM_HELDOUT_EVALUATION_EXECUTION_AMENDMENT_2026-08-24.md'
)
repair = (
    REPO / 'docs' / 'experiments'
    / 'PREREGISTRATION_SIM_HELDOUT_EVALUATION_EXECUTION_REPAIR_2026-08-24.md'
)
assert sha256_file(amendment) == AMENDMENT_SHA256
assert sha256_file(repair) == REPAIR_SHA256
pycache = WORK / 'livifuser_heldout_pycache'
pycache.mkdir(parents=True, exist_ok=True)
compile_environment = os.environ.copy()
compile_environment['PYTHONPYCACHEPREFIX'] = str(pycache)
subprocess.run(
    [sys.executable, '-m', 'compileall', '-q', str(REPO / 'src'), str(REPO / 'scripts')],
    check=True,
    env=compile_environment,
)
print(json.dumps({{'repository': str(REPO), 'cloud_bundle': verification}}, indent=2))
"""
    prepare = """
PLAN = WORK / 'livifuser_sim_heldout_data_plan_v1.json'
DATA_ROOT = WORK / 'livifuser_sim_heldout_data_v1'
subprocess.run(
    [
        sys.executable,
        str(REPO / 'scripts' / 'prepare_sim_heldout_data.py'),
        '--input-root',
        str(INPUT),
        '--work-root',
        str(DATA_ROOT),
        '--plan-output',
        str(PLAN),
    ],
    check=True,
)
plan = json.loads(PLAN.read_text(encoding='utf-8'))
assert (
    plan['episode_count'],
    plan['accepted_samples'],
    plan['windows_k8_h8'],
) == (110, 47326, 34503)
"""
    execute = f"""
result_anchors = [
    path
    for path in INPUT.rglob('RESULT_BUNDLE_MANIFEST.json')
    if path.is_file() and (path.parent / 'summary.json').is_file()
]
result_archives = [
    path
    for path in INPUT.rglob('livifuser_simulation_sweep_v1_results.zip')
    if path.is_file() and sha256_file(path) == '{RESULT_ARCHIVE_SHA256}'
]
assert len(result_anchors) + len(result_archives) == 1, (
    'attach exactly one livifuser-simulation-sweep-v1-results dataset; '
    f'expanded={{[str(path.parent) for path in result_anchors]}}, '
    f'archives={{[str(path) for path in result_archives]}}'
)
RESULT_SOURCE = (
    result_anchors[0].parent
    if result_anchors
    else result_archives[0]
)

score_manifests = [
    path
    for path in INPUT.rglob('SCORE_FREEZE_MANIFEST.json')
    if path.is_file() and sha256_file(path) == '{SCORE_MANIFEST_SHA256}'
]
score_archives = [
    path
    for path in INPUT.rglob('livifuser_sim_validation_score_freeze_v1_bundle.zip')
    if path.is_file() and sha256_file(path) == '{SCORE_ARCHIVE_SHA256}'
]
assert len(score_manifests) + len(score_archives) == 1, (
    'expected one frozen validation score source'
)
SCORE_SOURCE = score_manifests[0].parent if score_manifests else score_archives[0]

OUTPUT_ROOT = WORK / 'livifuser_sim_heldout_evaluation_v1_repair_eb19516b2d84'
BUNDLE = WORK / 'livifuser_sim_heldout_evaluation_v1_bundle.zip'
subprocess.run(
    [
        sys.executable,
        str(REPO / 'scripts' / 'run_sim_heldout_evaluation_kaggle.py'),
        '--data-plan',
        str(PLAN),
        '--results-source',
        str(RESULT_SOURCE),
        '--result-audit',
        str(REPO / 'artifacts' / 'simulation_sweep_v1_result_audit.json'),
        '--score-source',
        str(SCORE_SOURCE),
        '--score-audit',
        str(REPO / 'artifacts' / 'sim_validation_score_freeze_v1_audit.json'),
        '--config',
        str(config),
        '--output-root',
        str(OUTPUT_ROOT),
        '--bundle-output',
        str(BUNDLE),
        '--cuda-device',
        '0',
        '--cuda-device',
        '1',
    ],
    check=True,
)
assert BUNDLE.is_file()
print(json.dumps({{
    'download': str(BUNDLE),
    'size_bytes': BUNDLE.stat().st_size,
    'sha256': sha256_file(BUNDLE),
}}, indent=2))
"""
    download = """
from IPython.display import FileLink, display
display(FileLink(str(BUNDLE)))
"""
    return {
        "cells": [
            markdown_cell(
                """
# LiViFuser one-time simulation held-out evaluation

Run with a Kaggle T4 x2 accelerator and Internet disabled. Attach the expanded
held-out code, source handoff, DINO cache, frozen training results, and frozen
validation score datasets. Scientific metrics remain sealed in the returned
ZIP and are not printed by this notebook.
"""
            ),
            code_cell(preflight),
            code_cell(prepare),
            code_cell(execute),
            code_cell(download),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(notebook(), indent=1) + chr(10)
    output.write_text(payload, encoding="utf-8", newline=chr(10))
    print(
        json.dumps(
            {
                "notebook": str(output),
                "size_bytes": output.stat().st_size,
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest().upper(),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
