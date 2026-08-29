"""The shipped livifuser_nav subset must be importable on its own.

The RunPod handoff ships a curated subset of `src/livifuser_nav`. An earlier
build omitted `contracts.py`, which no shipped runtime module imports but which
`__init__.py` re-exports from -- so the package was unimportable on the pod and
`unpack_runpod_input_handoff.py` could not run from inside the bundle it was
shipped in. This checks the subset is import-closed without unpacking anything.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "livifuser_nav"


def _shipped_module_names() -> set[str]:
    # Read the member list as text rather than importing it: the point is to
    # check what the bundle *declares*, independently of what happens to be
    # importable in this interpreter.
    text = (ROOT / "src/livifuser_nav/runpod_handoff.py").read_text(encoding="utf-8")
    prefix = '"src/livifuser_nav/'
    names = set()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            names.add(stripped[len(prefix) :].split('"')[0].removesuffix(".py"))
    return names


def _sibling_imports(module: str) -> set[str]:
    tree = ast.parse((SOURCE / f"{module}.py").read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.level == 1:
                found.add(node.module.split(".")[0])
            elif node.module.startswith("livifuser_nav."):
                found.add(node.module.split(".")[1])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("livifuser_nav."):
                    found.add(alias.name.split(".")[1])
    return found


class RunpodHandoffSubsetTests(unittest.TestCase):
    def test_shipped_subset_is_import_closed(self) -> None:
        shipped = _shipped_module_names()
        self.assertIn("__init__", shipped)
        for module in sorted(shipped):
            for dependency in sorted(_sibling_imports(module)):
                self.assertIn(
                    dependency,
                    shipped,
                    f"{module}.py imports livifuser_nav.{dependency}, "
                    "which the RunPod handoff does not ship",
                )

    def test_contracts_is_shipped_because_init_reexports_it(self) -> None:
        self.assertIn("contracts", _shipped_module_names())


if __name__ == "__main__":
    unittest.main()


class LineEndingTests(unittest.TestCase):
    """CRLF in a shipped text file is a functional defect, not a style issue.

    `set -euo pipefail\r` aborts a shell script on its third line, and a CR
    terminates a shebang so the interpreter is never found. The bootstrap
    failed exactly this way after a 188 MB upload, because a patch script used
    `Path.write_text` on Windows without `newline=""`.
    """

    SUFFIXES = frozenset(
        {".sh", ".py", ".json", ".yaml", ".yml", ".xml", ".cfg", ".msg", ".md", ".toml"}
    )
    # Repository-root files count too: the builder guard first fired on
    # pyproject.toml, which an earlier version of this scan did not cover.
    ROOTS = ("scripts", "src/livifuser_nav", "config", "ros2_ws/src", "tests")
    ROOT_FILES = ("pyproject.toml", ".gitattributes", ".gitignore")

    def test_no_shipped_text_file_contains_a_carriage_return(self) -> None:
        offenders = []
        for root in self.ROOTS:
            for path in sorted((ROOT / root).rglob("*")):
                if (
                    path.is_file()
                    and path.suffix in self.SUFFIXES
                    and "__pycache__" not in path.parts
                    and path.read_bytes().count(b"\r")
                ):
                    offenders.append(path.relative_to(ROOT).as_posix())
        for name in self.ROOT_FILES:
            path = ROOT / name
            if path.is_file() and path.read_bytes().count(b"\r"):
                offenders.append(name)
        self.assertEqual(offenders, [], f"CRLF line endings in: {offenders}")

    def test_builder_rejects_a_carriage_return_member(self) -> None:
        from livifuser_nav.runpod_handoff import LINE_ENDING_SENSITIVE_SUFFIXES

        for suffix in (".sh", ".py", ".json", ".xml"):
            self.assertIn(suffix, LINE_ENDING_SENSITIVE_SUFFIXES)


class ShippedScriptTests(unittest.TestCase):
    """Every script the pod needs must actually be in the bundle.

    The Gate 6 sweep script was written locally, referenced by the run
    instructions, and never added to the handoff member list. The pod reported
    exit 127 -- command not found -- after a full rebuild and redeploy cycle.
    """

    def _shipped_scripts(self) -> list[str]:
        import re

        text = (ROOT / "src/livifuser_nav/runpod_handoff.py").read_text(encoding="utf-8")
        return re.findall(r'"(scripts/[a-z0-9_]+\.(?:py|sh))"', text)

    def test_every_shipped_script_exists_in_the_repository(self) -> None:
        missing = [name for name in self._shipped_scripts() if not (ROOT / name).is_file()]
        self.assertEqual(missing, [], f"handoff lists files that do not exist: {missing}")

    def test_scripts_the_pod_runs_are_shipped(self) -> None:
        shipped = set(self._shipped_scripts())
        for required in (
            "scripts/bootstrap_runpod_runtime.sh",
            "scripts/run_live_sim_development_episode.sh",
            "scripts/run_gate6_development_smoke.sh",
            "scripts/verify_recorded_input_determinism.py",
            "scripts/run_closed_loop_confirmatory_batch.py",
            "scripts/benchmark_splus_cuda_runtime.py",
            "scripts/wait_sim_terminal.py",
            "scripts/seal_runtime_attempt.py",
            "scripts/check_runpod_storage.py",
        ):
            self.assertIn(required, shipped)

    def test_gate6_sweep_only_calls_shipped_scripts(self) -> None:
        import re

        sweep = (ROOT / "scripts/run_gate6_development_smoke.sh").read_text(encoding="utf-8")
        shipped = set(self._shipped_scripts())
        for called in set(re.findall(r"(scripts/[a-z0-9_]+\.(?:py|sh))", sweep)):
            self.assertIn(called, shipped, f"{called} is called on the pod but not shipped")


class ShippedClosureTests(unittest.TestCase):
    """Whatever a shipped script imports or invokes must itself be shipped.

    Three separate omissions reached the pod before this existed: the Gate 6
    sweep script, the determinism harness, and the confirmatory batch runner and
    its plan module. Each was caught only by a remote failure after a full
    rebuild and redeploy. A hand-maintained list of "scripts the pod runs"
    cannot catch a name nobody thought to add, so this derives the requirement
    from the shipped files themselves.
    """

    def _shipped(self) -> set[str]:
        import re

        text = (ROOT / "src/livifuser_nav/runpod_handoff.py").read_text(encoding="utf-8")
        return set(re.findall(r'"((?:scripts|src/livifuser_nav)/[a-z0-9_]+\.(?:py|sh))"', text))

    def test_shipped_python_imports_only_shipped_modules(self) -> None:
        import ast

        shipped = self._shipped()
        missing: list[str] = []
        for relative in sorted(shipped):
            if not relative.endswith(".py"):
                continue
            tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                    "livifuser_nav."
                ):
                    names.append(node.module.split(".")[1])
                elif isinstance(node, ast.Import):
                    names += [
                        alias.name.split(".")[1]
                        for alias in node.names
                        if alias.name.startswith("livifuser_nav.")
                    ]
                for name in names:
                    target = f"src/livifuser_nav/{name}.py"
                    if target not in shipped:
                        missing.append(f"{relative} imports {name}, not shipped")
        self.assertEqual(missing, [], f"shipped set is not import-closed: {missing}")

    def test_shipped_scripts_invoke_only_shipped_scripts(self) -> None:
        import re

        shipped = self._shipped()
        missing: list[str] = []
        for relative in sorted(shipped):
            # Comments mention scripts without invoking them; a reference
            # in prose is not a dependency.
            source = (ROOT / relative).read_text(encoding="utf-8")
            text = chr(10).join(
                line
                for line in source.splitlines()
                if not line.lstrip().startswith("#")
            )
            for called in set(re.findall(r"(scripts/[a-z0-9_]+\.(?:py|sh))", text)):
                if called != relative and called not in shipped:
                    missing.append(f"{relative} invokes {called}, not shipped")
        self.assertEqual(missing, [], f"shipped scripts call unshipped scripts: {missing}")
