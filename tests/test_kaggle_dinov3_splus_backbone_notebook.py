from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class KaggleBackboneNotebookTests(unittest.TestCase):
    def test_generated_notebook_embeds_exact_core_and_compiles(self) -> None:
        notebook_path = (
            ROOT / "notebooks" / "kaggle_dinov3_splus_backbone_handoff_v1.ipynb"
        )
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        core = (ROOT / "src" / "livifuser_nav" / "backbone_handoff.py").read_text("utf-8")
        expected = hashlib.sha256(core.encode()).hexdigest().upper()
        self.assertEqual(notebook["metadata"]["livifuser"]["embedded_core_sha256"], expected)
        sources = [
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        ]
        self.assertIn("HF_TOKEN", "\n".join(sources))
        self.assertIn("model.safetensors", core)
        self.assertNotIn("dinov3_small_224.onnx", "\n".join(sources))
        for index, source in enumerate(sources):
            if source.startswith("%pip"):
                source = "\n".join(source.splitlines()[1:])
            compile(source, f"cell_{index}", "exec")


if __name__ == "__main__":
    unittest.main()
