"""The runtime handoff auditor must reject what it is meant to reject.

An audit that cannot fail is worse than no audit: it converts an unchecked
bundle into an apparently checked one. Every test here tampers with a sealed
bundle and asserts the auditor notices, alongside the happy path.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path

from livifuser_nav.runtime_handoff import (
    BUNDLE_ROOT,
    COMPLETION_NAME,
    EXPECTED_POLICY_IDENTITIES,
    MANIFEST_NAME,
    REQUIRED_CODE,
    REQUIRED_CONFIG,
    REQUIRED_EVIDENCE,
    _canonical_self_hash,
    audit_runtime_handoff,
    build_runtime_handoff,
)

ROOT = Path(__file__).resolve().parents[1]


def _stub_evidence(directory: Path) -> None:
    """Minimal evidence that satisfies the auditor's content checks."""

    (directory / "cuda_route_benchmark_v1.json").write_text(
        json.dumps(
            {
                "status": "CUDA_ROUTE_ACCEPTED",
                "parity": {"pass": True},
                "timing": {"pass": True},
                "forbidden_inputs_used": {"heldout": False, "confirmatory": False},
            }
        )
    )
    (directory / "recorded_input_determinism_v1.json").write_text(
        json.dumps(
            {
                "deterministic": True,
                "policy_identities": EXPECTED_POLICY_IDENTITIES,
                "device": "cuda:0",
                "forbidden_inputs_used": {"heldout": False, "confirmatory": False},
            }
        )
    )


def _stub_zip(path: Path, marker: str) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("payload.txt", marker)


class RuntimeHandoffTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        evidence = self.tmp / "evidence"
        evidence.mkdir()
        _stub_evidence(evidence)
        backbone = self.tmp / "backbone.zip"
        policies = self.tmp / "policies.zip"
        _stub_zip(backbone, "backbone")
        _stub_zip(policies, "policies")
        self.bundle = self.tmp / "runtime.zip"
        self.report = build_runtime_handoff(
            ROOT,
            evidence,
            self.bundle,
            backbone_bundle=backbone,
            policy_payload=policies,
        )

    def _rebuild_with(self, mutate) -> Path:
        """Copy the sealed bundle, apply a mutation, return the new path."""

        target = self.tmp / "tampered.zip"
        target.unlink(missing_ok=True)
        with zipfile.ZipFile(self.bundle) as source:
            entries = {name: source.read(name) for name in source.namelist()}
        mutate(entries)
        with zipfile.ZipFile(target, "w") as archive:
            for name, payload in entries.items():
                archive.writestr(name, payload)
        return target


class RuntimeHandoffAuditTests(RuntimeHandoffTestCase):
    def test_a_sealed_bundle_audits_clean(self) -> None:
        report = audit_runtime_handoff(self.bundle)
        self.assertEqual(report["findings"], [])
        self.assertEqual(report["status"], "audit_pass")
        self.assertFalse(report["confirmatory_launch_authorized"])

    def test_every_required_member_is_present(self) -> None:
        with zipfile.ZipFile(self.bundle) as archive:
            names = {n[len(BUNDLE_ROOT) + 1 :] for n in archive.namelist()}
        for relative in REQUIRED_CODE + REQUIRED_CONFIG + REQUIRED_EVIDENCE:
            self.assertIn(relative, names)

    def test_a_modified_member_is_caught(self) -> None:
        def mutate(entries):
            key = f"{BUNDLE_ROOT}/scripts/wait_sim_terminal.py"
            entries[key] = entries[key] + b"\n# tampered\n"

        report = audit_runtime_handoff(self._rebuild_with(mutate))
        self.assertEqual(report["status"], "audit_fail")
        self.assertTrue(any("hash mismatch" in f for f in report["findings"]))

    def test_a_removed_member_is_caught(self) -> None:
        def mutate(entries):
            del entries[f"{BUNDLE_ROOT}/src/livifuser_nav/live_runtime.py"]

        report = audit_runtime_handoff(self._rebuild_with(mutate))
        self.assertEqual(report["status"], "audit_fail")
        self.assertTrue(any("missing member" in f for f in report["findings"]))

    def test_an_extra_member_is_caught(self) -> None:
        def mutate(entries):
            entries[f"{BUNDLE_ROOT}/extra_payload.bin"] = b"unexpected"

        report = audit_runtime_handoff(self._rebuild_with(mutate))
        self.assertEqual(report["status"], "audit_fail")
        self.assertTrue(any("not in the manifest" in f for f in report["findings"]))

    def test_a_tampered_manifest_is_caught(self) -> None:
        def mutate(entries):
            manifest = json.loads(entries[MANIFEST_NAME])
            manifest["member_count"] = 1
            entries[MANIFEST_NAME] = json.dumps(manifest).encode()

        with self.assertRaises(ValueError):
            audit_runtime_handoff(self._rebuild_with(mutate))

    def test_a_completion_marker_that_stops_binding_the_manifest_is_caught(self) -> None:
        def mutate(entries):
            completion = json.loads(entries[COMPLETION_NAME])
            completion["manifest_file_sha256"] = "0" * 64
            entries[COMPLETION_NAME] = json.dumps(completion).encode()

        with self.assertRaises(ValueError):
            audit_runtime_handoff(self._rebuild_with(mutate))


def _reseal(entries: dict, member: str, payload: bytes) -> None:
    """Replace a member and restore the manifest so only content is wrong.

    Without this the manifest self-hash fails first and the content check under
    test never runs.
    """

    import hashlib

    entries[f"{BUNDLE_ROOT}/{member}"] = payload
    manifest = json.loads(entries[MANIFEST_NAME])
    for record in manifest["members"]:
        if record["name"] == member:
            record["sha256"] = hashlib.sha256(payload).hexdigest().upper()
            record["size_bytes"] = len(payload)
    manifest.pop("manifest_sha256_excludes_self", None)
    manifest["manifest_sha256_excludes_self"] = _canonical_self_hash(
        manifest, "manifest_sha256_excludes_self"
    )
    manifest_bytes = json.dumps(manifest, sort_keys=True, indent=2).encode() + b"\n"
    entries[MANIFEST_NAME] = manifest_bytes

    completion = json.loads(entries[COMPLETION_NAME])
    completion["manifest_file_sha256"] = hashlib.sha256(manifest_bytes).hexdigest().upper()
    completion["manifest_sha256_excludes_self"] = manifest["manifest_sha256_excludes_self"]
    completion.pop("completion_sha256_excludes_self", None)
    completion["completion_sha256_excludes_self"] = _canonical_self_hash(
        completion, "completion_sha256_excludes_self"
    )
    entries[COMPLETION_NAME] = json.dumps(completion, sort_keys=True, indent=2).encode() + b"\n"


class RuntimeHandoffScopeTests(RuntimeHandoffTestCase):
    def test_cpu_determinism_evidence_is_rejected(self) -> None:
        # The gate requires the execution route. A CPU pass does not establish
        # determinism on CUDA, where cuBLAS needs a workspace configuration.
        member = "evidence/recorded_input_determinism_v1.json"

        def mutate(entries):
            evidence = json.loads(entries[f"{BUNDLE_ROOT}/{member}"])
            evidence["device"] = "cpu"
            _reseal(entries, member, json.dumps(evidence).encode())

        report = audit_runtime_handoff(self._rebuild_with(mutate))
        self.assertTrue(any("CPU" in f for f in report["findings"]))

    def test_evidence_admitting_a_forbidden_input_is_rejected(self) -> None:
        member = "evidence/cuda_route_benchmark_v1.json"

        def mutate(entries):
            evidence = json.loads(entries[f"{BUNDLE_ROOT}/{member}"])
            evidence["forbidden_inputs_used"] = {"heldout": True, "confirmatory": False}
            _reseal(entries, member, json.dumps(evidence).encode())

        report = audit_runtime_handoff(self._rebuild_with(mutate))
        self.assertTrue(any("forbidden input" in f for f in report["findings"]))

    def test_a_cached_feature_bundle_cannot_be_shipped(self) -> None:
        def mutate(entries):
            entries[f"{BUNDLE_ROOT}/artifacts/livifuser_dinov3_splus_cache_v2_bundle.zip"] = b"x"

        with self.assertRaises(ValueError):
            audit_runtime_handoff(self._rebuild_with(mutate))

    def test_the_builder_refuses_to_overwrite(self) -> None:
        evidence = self.tmp / "evidence"
        with self.assertRaises(FileExistsError):
            build_runtime_handoff(
                ROOT,
                evidence,
                self.bundle,
                backbone_bundle=self.tmp / "backbone.zip",
                policy_payload=self.tmp / "policies.zip",
            )


if __name__ == "__main__":
    unittest.main()
