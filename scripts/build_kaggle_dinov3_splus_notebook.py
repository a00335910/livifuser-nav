"""Build the frozen DINOv3 S+/16 train/validation cache notebook."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = ROOT / "scripts" / "kaggle_cache_dinov3_splus_core.py"
NOTEBOOK_PATH = ROOT / "notebooks" / "kaggle_dinov3_splus_cache_train_val_v1.ipynb"


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


def build_notebook(core_source: str) -> dict[str, object]:
    return {
        "cells": [
            markdown_cell(
                dedent(
                    """\
                    # LiViFuser frozen DINOv3 ViT-S+/16 feature cache

                    This notebook extracts the preregistered frozen visual features for the
                    150-episode train/validation handoff. It does **not** train a policy, fit
                    Mahalanobis statistics, or inspect confirmatory test outcomes.

                    Before running:

                    1. Enable a Kaggle GPU accelerator and Internet access. Two T4 GPUs are
                       supported, but one GPU also works.
                    2. Attach the dataset containing `handoff_manifest.json`. Kaggle may mount each
                       source ZIP as an extracted directory; both layouts are supported.
                    3. Accept access to Meta's gated
                       `facebook/dinov3-vits16plus-pretrain-lvd1689m` model on Hugging Face.
                    4. Add a private Kaggle secret named `HF_TOKEN` with read access. Never paste
                       the token into a cell.

                    The notebook refuses any model revision, checkpoint hash, handoff hash, tensor
                    shape, or preprocessing contract that differs from the frozen values. It keeps
                    the full camera field of view using the project letterbox transform:
                    320×240 → 224×168 with 28 normalized-zero rows above and below.

                    Outputs are restartable world-level ZIP shards in
                    `/kaggle/working/livifuser_dinov3_splus_cache_v2`. If a run is interrupted,
                    rerun the cache cell in the same Kaggle session; completed episodes and shards
                    are verified before being reused.
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
                        [
                            sys.executable,
                            "-m",
                            "pip",
                            "install",
                            "--quiet",
                            *PINNED_PACKAGES,
                        ]
                    )
                    print("Installed:", ", ".join(PINNED_PACKAGES))
                    """
                ).strip()
            ),
            markdown_cell(
                "## Frozen implementation\n\n"
                "This cell is generated from the tracked core implementation. Do not edit its "
                "model identity, preprocessing, or output-shape constants."
            ),
            code_cell(core_source),
            markdown_cell("## Discover and verify the uploaded 150-episode handoff"),
            code_cell(
                dedent(
                    """\
                    from pathlib import Path

                    INPUT_ROOT = Path("/kaggle/input")
                    OUTPUT_ROOT = Path(
                        "/kaggle/working/livifuser_dinov3_splus_cache_v2"
                    )
                    WORK_ROOT = Path(
                        "/kaggle/working/livifuser_dinov3_cache_work_v2"
                    )
                    SCRATCH_ROOT = Path(
                        "/kaggle/working/livifuser_dinov3_rgb_scratch_v2"
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
                "## Extract and seal all feature shards\n\n"
                "This processes only the frozen train/validation handoff. Existing complete "
                "shards are hash-verified and skipped; unfinished episode work resumes safely "
                "within the current Kaggle session."
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
            markdown_cell("## Final deep integrity verification"),
            code_cell(
                "verification = verify_cache_output(OUTPUT_ROOT)\n"
                "print(json.dumps(verification, indent=2))\n"
                'print("Kaggle output ready at:", OUTPUT_ROOT)'
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
    core_source = CORE_PATH.read_text(encoding="utf-8")
    notebook = build_notebook(core_source)
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(NOTEBOOK_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
