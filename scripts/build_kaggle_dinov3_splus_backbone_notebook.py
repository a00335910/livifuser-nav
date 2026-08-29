#!/usr/bin/env python3
"""Generate the deterministic Kaggle handoff for the official runtime backbone."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "src" / "livifuser_nav" / "backbone_handoff.py"
OUTPUT = ROOT / "notebooks" / "kaggle_dinov3_splus_backbone_handoff_v1.ipynb"


def code_cell(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def markdown_cell(source: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def main() -> None:
    core = CORE.read_text(encoding="utf-8")
    core_sha = hashlib.sha256(core.encode("utf-8")).hexdigest().upper()
    notebook = {
        "cells": [
            markdown_cell(
                "# LiViFuser official DINOv3 S+/16 backbone handoff v1\n\n"
                "Accept access to the gated Hugging Face model in your browser, add a private "
                "Kaggle secret named `HF_TOKEN`, enable Internet, and run all cells. This notebook "
                "downloads no policy data and performs no training or held-out inference.\n"
            ),
            code_cell(
                "%pip install -q huggingface_hub==0.34.4\n"
                "from kaggle_secrets import UserSecretsClient\n"
                "HF_TOKEN = UserSecretsClient().get_secret('HF_TOKEN')\n"
                "if not HF_TOKEN or not HF_TOKEN.strip():\n"
                "    raise RuntimeError('HF_TOKEN is missing or empty')\n"
            ),
            code_cell(
                f"# Embedded source SHA-256: {core_sha}\n"
                f"CORE_SOURCE = {core!r}\n"
                "namespace = {'__name__': 'livifuser_backbone_handoff_embedded'}\n"
                "exec(compile(CORE_SOURCE, 'backbone_handoff.py', 'exec'), namespace)\n"
            ),
            code_cell(
                "from pathlib import Path\n"
                "work = Path('/kaggle/working/livifuser_dinov3_splus_backbone_v1')\n"
                "cache = work / 'hf_cache'\n"
                "out = work / 'output'\n"
                "out.mkdir(parents=True, exist_ok=True)\n"
                "snapshot = namespace['download_snapshot'](token=HF_TOKEN, cache_dir=cache)\n"
                "bundle = out / namespace['BUNDLE_FILENAME']\n"
                "report = namespace['seal_snapshot'](snapshot, bundle)\n"
                "del HF_TOKEN\n"
                "print('BUNDLE', bundle)\n"
                "print('SIZE', report['bundle_size_bytes'])\n"
                "print('SHA256', report['bundle_sha256'])\n"
                "print('WEIGHTS_SHA256', report['weights_sha256'])\n"
                "display(bundle)\n"
            ),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
            "livifuser": {"embedded_core_sha256": core_sha, "schema_version": "1.0.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()

