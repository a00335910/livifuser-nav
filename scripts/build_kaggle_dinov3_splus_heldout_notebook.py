"""Build the frozen DINOv3 S+/16 held-out feature-cache notebook."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = ROOT / "scripts" / "kaggle_cache_dinov3_splus_core.py"
NOTEBOOK_PATH = ROOT / "notebooks" / "kaggle_dinov3_splus_cache_heldout_v1.ipynb"
HANDOFF_SHA256 = "3F48A7E54A1596947A469B59B8D63EE96FD29294C7BD57EFAC924736A984492C"
HANDOFF_NAME = "livifuser_confirmatory_v3_heldout_v1"
CACHE_NAME = "livifuser_dinov3_vits16plus_heldout_cache_v1"
EXPECTED_EPISODES = 110
EXPECTED_SAMPLES = 47_326
TRAIN_VAL_HANDOFF_SHA256 = "AB24252411EEF448BC0D853B0C9147AF184F0A1CC14D72BA39876BF179A92C6F"


def markdown_cell(source: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code_cell(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def specialized_core(core_source: str) -> str:
    replacements = (
        (
            f'EXPECTED_HANDOFF_SHA256 = "{TRAIN_VAL_HANDOFF_SHA256}"',
            f'EXPECTED_HANDOFF_SHA256 = "{HANDOFF_SHA256}"',
        ),
        (
            'EXPECTED_HANDOFF_NAME = "livifuser_confirmatory_v3_train_val_v1"',
            f'EXPECTED_HANDOFF_NAME = "{HANDOFF_NAME}"',
        ),
        (
            'CACHE_NAME = "livifuser_dinov3_vits16plus_train_val_cache_v2"',
            f'CACHE_NAME = "{CACHE_NAME}"',
        ),
        (
            'manifest["audit"]["episodes"] == 150',
            'manifest["audit"]["episodes"] == 110',
        ),
        (
            'manifest["audit"]["by_split"] == {"train": 120, "val_id": 30}',
            'manifest["audit"]["by_split"] == {"test_id": 30, "test_ood": 80}',
        ),
        (
            'manifest["audit"]["accepted_samples"] == 69_253',
            'manifest["audit"]["accepted_samples"] == 47_326',
        ),
        (
            "list(range(150))",
            "list(range(150, 260))",
        ),
        (
            "Counter(train=120, val_id=30)",
            "Counter(test_id=30, test_ood=80)",
        ),
        (
            'master["audit"]["episodes"] == 150',
            'master["audit"]["episodes"] == 110',
        ),
        (
            'master["audit"]["accepted_samples"] == 69_253',
            'master["audit"]["accepted_samples"] == 47_326',
        ),
        (
            'master["audit"]["by_split"] == {"train": 120, "val_id": 30}',
            'master["audit"]["by_split"] == {"test_id": 30, "test_ood": 80}',
        ),
    )
    result = core_source
    twice = {
        'manifest["audit"]["episodes"] == 150',
        'manifest["audit"]["accepted_samples"] == 69_253',
    }
    for old, new in replacements:
        count = result.count(old)
        expected_count = 2 if old in twice else 1
        if count != expected_count:
            raise ValueError(
                f"expected {expected_count} frozen core fragments, found {count}: {old}"
            )
        result = result.replace(old, new)
    return result


def build_notebook(core_source: str) -> dict[str, object]:
    return {
        "cells": [
            markdown_cell(
                dedent(
                    """\
                    # LiViFuser frozen DINOv3 ViT-S+/16 held-out feature cache

                    This notebook extracts frozen visual features for the separately packaged
                    110-episode held-out handoff: 30 test-ID episodes and 80 test-OOD episodes.
                    It does **not** train a policy, fit normalization or Gaussian/Mahalanobis
                    statistics, select checkpoints, tune thresholds, or evaluate policy outcomes.

                    Before running:

                    1. Enable a Kaggle GPU accelerator and Internet access. Two T4 GPUs are
                       supported, but one GPU also works.
                    2. Attach the dataset containing the ten held-out ZIP shards,
                       `handoff_manifest.json`, `README.txt`, and all `.sha256` sidecars.
                       Kaggle may mount ZIPs as extracted directories; both layouts work.
                    3. Accept access to Meta's gated
                       `facebook/dinov3-vits16plus-pretrain-lvd1689m` model on Hugging Face.
                    4. Add a private Kaggle secret named `HF_TOKEN`. Never paste the token into
                       a cell.

                    The notebook pins the same model revision, weight hash, full-FOV letterbox
                    preprocessing, float32 backbone compute, and feature shapes used for the
                    accepted train/validation cache. Held-out output paths and cache identity are
                    distinct. The final cell creates one downloadable ZIP bundle.
                    """
                ).strip()
            ),
            code_cell(
                dedent(
                    """\
                    import subprocess
                    import sys

                    PINNED_PACKAGES = [
                        "transformers==4.56.0",
                        "huggingface_hub==0.34.4",
                        "safetensors==0.6.2",
                    ]
                    subprocess.check_call(
                        [sys.executable, "-m", "pip", "install", "--quiet", *PINNED_PACKAGES]
                    )
                    print("Installed:", ", ".join(PINNED_PACKAGES))
                    """
                ).strip()
            ),
            markdown_cell(
                "## Frozen held-out implementation\n\n"
                "Generated from the accepted train/validation cache implementation with only "
                "the handoff identity, held-out counts, ordinal range, and cache name replaced."
            ),
            code_cell(specialized_core(core_source)),
            markdown_cell("## Discover and verify the uploaded 110-episode handoff"),
            code_cell(
                dedent(
                    """\
                    from pathlib import Path

                    INPUT_ROOT = Path("/kaggle/input")
                    OUTPUT_ROOT = Path(
                        "/kaggle/working/livifuser_dinov3_splus_heldout_cache_v1"
                    )
                    WORK_ROOT = Path(
                        "/kaggle/working/livifuser_dinov3_heldout_work_v1"
                    )
                    SCRATCH_ROOT = Path(
                        "/kaggle/working/livifuser_dinov3_heldout_rgb_scratch_v1"
                    )
                    MODEL_CACHE = Path("/kaggle/working/huggingface_model_cache")
                    BATCH_SIZE = 64

                    handoff = discover_handoff(INPUT_ROOT)
                    print(json.dumps({
                        "manifest": str(handoff["manifest_path"]),
                        "manifest_sha256": EXPECTED_HANDOFF_SHA256,
                        "episodes": len(handoff["episodes"]),
                        "accepted_samples": sum(
                            source["episode"]["accepted_samples"]
                            for source in handoff["episodes"]
                        ),
                        "archives": len(handoff["archives"]),
                        "batch_size": BATCH_SIZE,
                    }, indent=2))
                    """
                ).strip()
            ),
            markdown_cell("## Authenticate and load the exact frozen backbone"),
            code_cell(
                dedent(
                    """\
                    from kaggle_secrets import UserSecretsClient

                    try:
                        HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
                    except Exception as error:
                        raise RuntimeError(
                            "Add a private Kaggle secret named HF_TOKEN after accepting "
                            "the gated DINOv3 model license on Hugging Face."
                        ) from error

                    model, backbone_contract_sha256, backbone_contract = (
                        load_frozen_backbone(HF_TOKEN, MODEL_CACHE, OUTPUT_ROOT)
                    )
                    del HF_TOKEN
                    print(json.dumps({
                        "model": MODEL_ID,
                        "revision": MODEL_REVISION,
                        "weights_sha256": MODEL_SAFETENSORS_SHA256,
                        "contract_file_sha256": backbone_contract_sha256,
                        "cuda_devices": backbone_contract["inference"]["cuda_devices"],
                        "parameter_count": backbone_contract["backbone"]["parameter_count"],
                    }, indent=2))
                    """
                ).strip()
            ),
            markdown_cell("## Smoke-test preprocessing and feature shapes"),
            code_cell(
                "smoke = smoke_backbone(handoff, model, SCRATCH_ROOT)\n"
                "print(json.dumps(smoke, indent=2))"
            ),
            markdown_cell(
                "## Extract and seal all held-out feature shards\n\n"
                "Completed episode and world shards are hash-verified and skipped on rerun. "
                "No training statistic is calculated here."
            ),
            code_cell(
                dedent(
                    """\
                    cache_manifest = run_cache(
                        handoff=handoff,
                        model=model,
                        backbone_contract_sha256=backbone_contract_sha256,
                        output_root=OUTPUT_ROOT,
                        work_root=WORK_ROOT,
                        scratch_root=SCRATCH_ROOT,
                        batch_size=BATCH_SIZE,
                    )
                    print(json.dumps(cache_manifest["audit"], indent=2))
                    """
                ).strip()
            ),
            markdown_cell("## Final integrity verification"),
            code_cell(
                "verification = verify_cache_output(OUTPUT_ROOT)\n"
                "print(json.dumps(verification, indent=2))\n"
                'print("Kaggle output ready at:", OUTPUT_ROOT)'
            ),
            markdown_cell("## Create one downloadable transport bundle"),
            code_cell(
                dedent(
                    """\
                    BUNDLE_PATH = Path(
                        "/kaggle/working/livifuser_dinov3_splus_heldout_cache_v1_bundle.zip"
                    )
                    BUNDLE_PARTIAL = BUNDLE_PATH.with_suffix(".zip.partial")
                    if BUNDLE_PARTIAL.exists():
                        BUNDLE_PARTIAL.unlink()
                    if BUNDLE_PATH.exists():
                        BUNDLE_PATH.unlink()
                    with zipfile.ZipFile(BUNDLE_PARTIAL, "w", allowZip64=True) as archive:
                        for path in sorted(
                            item for item in OUTPUT_ROOT.rglob("*") if item.is_file()
                        ):
                            member = path.relative_to(OUTPUT_ROOT).as_posix()
                            write_file_to_zip(archive, member, path)
                    os.replace(BUNDLE_PARTIAL, BUNDLE_PATH)
                    print(json.dumps({
                        "bundle": str(BUNDLE_PATH),
                        "size_bytes": BUNDLE_PATH.stat().st_size,
                        "size_gib": round(BUNDLE_PATH.stat().st_size / 1024**3, 3),
                        "sha256": sha256_file(BUNDLE_PATH),
                    }, indent=2))
                    """
                ).strip()
            ),
        ],
        "metadata": {
            "accelerator": "GPU",
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    notebook = build_notebook(CORE_PATH.read_text(encoding="utf-8"))
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(NOTEBOOK_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
