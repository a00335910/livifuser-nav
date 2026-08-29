import importlib.util
import tempfile
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "run_pilot5_cv.py"
SPEC = importlib.util.spec_from_file_location("run_pilot5_cv", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TestPilot5Cv(unittest.TestCase):
    def test_fold_command_carries_isolated_device_and_exact_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            command = MODULE.fold_command(
                Path("config.json"),
                MODULE.EPISODES[0],
                Path(temporary) / "fold",
                "cuda:0",
            )
        self.assertEqual(command[command.index("--device") + 1], "cuda:0")
        self.assertEqual(command.count("--val-export"), 1)
        self.assertEqual(command.count("--val-cache"), 1)
        self.assertEqual(command.count("--train-export"), 4)
        self.assertEqual(command.count("--train-cache"), 4)


if __name__ == "__main__":
    unittest.main()
