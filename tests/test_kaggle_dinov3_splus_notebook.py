from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from livifuser_nav.learning_data import preprocess_rgb

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "scripts" / "kaggle_cache_dinov3_splus_core.py"
BUILDER = ROOT / "scripts" / "build_kaggle_dinov3_splus_notebook.py"
NOTEBOOK = ROOT / "notebooks" / "kaggle_dinov3_splus_cache_train_val_v1.ipynb"


class KaggleDinov3SplusNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.core_source = CORE.read_text(encoding="utf-8")
        cls.notebook_bytes = NOTEBOOK.read_bytes()
        cls.notebook = json.loads(cls.notebook_bytes.decode("utf-8"))
        cls.all_source = "\n".join(cell["source"] for cell in cls.notebook["cells"])

    def test_all_code_cells_compile(self) -> None:
        for index, cell in enumerate(self.notebook["cells"]):
            if cell["cell_type"] == "code":
                compile(cell["source"], f"notebook-cell-{index}", "exec")

    def test_embedded_core_is_exact(self) -> None:
        embedded = [
            cell["source"]
            for cell in self.notebook["cells"]
            if cell["cell_type"] == "code"
            and cell["source"].startswith('"""Self-contained Kaggle helpers')
        ]
        self.assertEqual(embedded, [self.core_source])

    def test_frozen_model_identity_is_present(self) -> None:
        expected = (
            "facebook/dinov3-vits16plus-pretrain-lvd1689m",
            "c93d816fc9e567563bc068f01475bec89cc634a6",
            "208146E499DACE99E4C9376DDB8A26F77D64C31C46C4DC4B86FF8BC63B0235E2",
            "AB24252411EEF448BC0D853B0C9147AF184F0A1CC14D72BA39876BF179A92C6F",
        )
        for value in expected:
            self.assertIn(value, self.all_source)

    def test_self_hash_uses_packager_canonical_json(self) -> None:
        self.assertIn(
            "return sha256_bytes(canonical_bytes(payload))",
            self.core_source,
        )

    def test_temporary_onnx_baseline_is_not_used(self) -> None:
        lowered = self.all_source.lower()
        self.assertNotIn("dinov3_small_224.onnx", lowered)
        self.assertNotIn("onnxruntime", lowered)

    def test_secret_is_retrieved_without_literal_token(self) -> None:
        self.assertIn('get_secret("HF_TOKEN")', self.all_source)
        self.assertIsNone(re.search(r"hf_[A-Za-z0-9]{20,}", self.all_source))

    def test_handoff_and_feature_contract_are_locked(self) -> None:
        for value in (
            "69_253",
            "Counter(train=120, val_id=30)",
            "shape=(row_count, 49, 384)",
            "shape=(row_count, 384)",
            "full_fov_letterbox_320x240_to_224x224_imagenet_pillow_bicubic_v1",
        ):
            self.assertIn(value, self.all_source)

    def test_notebook_preprocessing_matches_project_implementation(self) -> None:
        tree = ast.parse(self.core_source)
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "preprocess_rgb_batch"
        )
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__",
                    names=[ast.alias(name="annotations")],
                    level=0,
                ),
                function,
            ],
            type_ignores=[],
        )
        namespace = {
            "np": np,
            "Image": Image,
            "RGB_MEAN": np.asarray([0.485, 0.456, 0.406], dtype=np.float32),
            "RGB_STD": np.asarray([0.229, 0.224, 0.225], dtype=np.float32),
            "require": lambda condition, message: None if condition else self.fail(message),
            "torch": SimpleNamespace(from_numpy=lambda value: value),
        }
        exec(
            compile(ast.fix_missing_locations(module), "preprocess-extract", "exec"),
            namespace,
        )
        random = np.random.default_rng(20260823)
        images = random.integers(0, 256, size=(3, 240, 320, 3), dtype=np.uint8)
        observed = namespace["preprocess_rgb_batch"](images)
        expected = np.stack([preprocess_rgb(image) for image in images])
        np.testing.assert_array_equal(observed, expected)

    def test_output_shards_cannot_be_confused_with_source_shards(self) -> None:
        self.assertIn(".dinov3_vits16plus_cache.zip", self.all_source)

    def test_kaggle_extracted_shards_are_supported(self) -> None:
        self.assertIn('source_layout = "zip" if archive_matches else', self.core_source)
        self.assertIn('"kaggle_extracted_directory"', self.core_source)
        self.assertIn("extracted member set drifted", self.core_source)

    def test_t4_inference_uses_finite_fp32_compute(self) -> None:
        self.assertIn('"compute_dtype": "float32"', self.core_source)
        self.assertIn('"autocast": "disabled after T4 smoke', self.core_source)
        self.assertNotIn("torch.autocast", self.core_source)
        self.assertIn("dtype=torch.float32, non_blocking=True", self.core_source)

    def test_notebook_generation_is_deterministic(self) -> None:
        subprocess.run([sys.executable, str(BUILDER)], cwd=ROOT, check=True)
        self.assertEqual(NOTEBOOK.read_bytes(), self.notebook_bytes)


if __name__ == "__main__":
    unittest.main()
