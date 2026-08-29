from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "scripts" / "kaggle_cache_dinov3_splus_core.py"
BUILDER = ROOT / "scripts" / "build_kaggle_dinov3_splus_heldout_notebook.py"
NOTEBOOK = ROOT / "notebooks" / "kaggle_dinov3_splus_cache_heldout_v1.ipynb"
SPEC = importlib.util.spec_from_file_location("heldout_notebook_builder", BUILDER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class KaggleDinov3SplusHeldoutNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.notebook_bytes = NOTEBOOK.read_bytes()
        cls.notebook = json.loads(cls.notebook_bytes.decode("utf-8"))
        cls.all_source = "\n".join(cell["source"] for cell in cls.notebook["cells"])
        cls.specialized = MODULE.specialized_core(CORE.read_text(encoding="utf-8"))

    def test_all_code_cells_compile(self) -> None:
        for index, cell in enumerate(self.notebook["cells"]):
            if cell["cell_type"] == "code":
                compile(cell["source"], f"heldout-notebook-cell-{index}", "exec")

    def test_embedded_specialized_core_is_exact(self) -> None:
        embedded = [
            cell["source"]
            for cell in self.notebook["cells"]
            if cell["cell_type"] == "code"
            and cell["source"].startswith('"""Self-contained Kaggle helpers')
        ]
        self.assertEqual(embedded, [self.specialized])

    def test_heldout_identity_and_boundaries_are_locked(self) -> None:
        for value in (
            MODULE.HANDOFF_SHA256,
            MODULE.HANDOFF_NAME,
            MODULE.CACHE_NAME,
            "47_326",
            "list(range(150, 260))",
            "Counter(test_id=30, test_ood=80)",
        ):
            self.assertIn(value, self.all_source)

    def test_backbone_and_preprocessing_match_train_val(self) -> None:
        for value in (
            "facebook/dinov3-vits16plus-pretrain-lvd1689m",
            "c93d816fc9e567563bc068f01475bec89cc634a6",
            "208146E499DACE99E4C9376DDB8A26F77D64C31C46C4DC4B86FF8BC63B0235E2",
            "full_fov_letterbox_320x240_to_224x224_imagenet_pillow_bicubic_v1",
            '"compute_dtype": "float32"',
        ):
            self.assertIn(value, self.all_source)

    def test_output_and_bundle_are_distinct(self) -> None:
        self.assertIn("livifuser_dinov3_splus_heldout_cache_v1", self.all_source)
        self.assertIn("heldout_cache_v1_bundle.zip", self.all_source)
        self.assertNotIn('/kaggle/working/livifuser_dinov3_splus_cache_v2"', self.all_source)

    def test_notebook_forbids_heldout_statistic_fitting(self) -> None:
        markdown = "\n".join(
            cell["source"] for cell in self.notebook["cells"] if cell["cell_type"] == "markdown"
        )
        self.assertIn("does **not** train", markdown)
        self.assertIn("Gaussian/Mahalanobis", markdown)
        self.assertIn("No training statistic is calculated", markdown)

    def test_temporary_onnx_baseline_and_literal_token_are_absent(self) -> None:
        self.assertNotIn("dinov3_small_224.onnx", self.all_source.lower())
        self.assertNotIn("onnxruntime", self.all_source.lower())
        self.assertIn('get_secret("HF_TOKEN")', self.all_source)

    def test_notebook_generation_is_deterministic(self) -> None:
        subprocess.run([sys.executable, str(BUILDER)], cwd=ROOT, check=True)
        self.assertEqual(NOTEBOOK.read_bytes(), self.notebook_bytes)


if __name__ == "__main__":
    unittest.main()
