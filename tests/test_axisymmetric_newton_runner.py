#!/usr/bin/env python3
"""Check that rejected benchmark reruns preserve saved provenance and samples."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                       / "benchmarks" / "axisymmetric_newton"))
import run as newton_run


class NewtonRunnerTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.exe = self.root / "cumes"
        self.exe.write_bytes(b"benchmark executable")
        input_path = self.root / "input.json"
        input_path.write_text(json.dumps({
            "ntor": 0, "lfreeb": False, "ns_array": [5],
            "niter_array": [10], "ftol_array": [1e-6],
        }))
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(json.dumps({"cases": [{
            "id": "case", "input": input_path.name,
            "sha256": newton_run.digest(input_path),
        }]}))
        self.out = self.root / "results"
        process_patch = patch.object(
            newton_run.subprocess, "run",
            return_value=subprocess.CompletedProcess([], 1))
        self.process = process_patch.start()
        self.addCleanup(process_patch.stop)
        telemetry_patch = patch.object(newton_run, "telemetry", return_value="")
        telemetry_patch.start()
        self.addCleanup(telemetry_patch.stop)

    def invoke(self, *options):
        argv = ["run.py", "--exe", str(self.exe), "--manifest",
                str(self.manifest), "--out", str(self.out), *options]
        with patch.object(sys, "argv", argv), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            newton_run.main()

    def snapshot(self):
        return {path.name: (path.read_bytes(), path.stat().st_mtime_ns)
                for path in self.out.iterdir()}

    def assert_rejected_without_writes(self, *options, error=SystemExit):
        before = self.snapshot()
        self.process.reset_mock()
        with self.assertRaises(error):
            self.invoke(*options)
        self.assertEqual(self.snapshot(), before)
        self.process.assert_not_called()

    def test_fresh_and_empty_output(self):
        for exists in (False, True):
            with self.subTest(existing_empty_directory=exists):
                self.out = self.root / str(exists)
                if exists:
                    self.out.mkdir()
                self.process.reset_mock()
                self.invoke()
                self.assertEqual(self.process.call_count, 2)
                protocol = json.loads((self.out / "protocol.json").read_text())
                self.assertEqual(protocol["binary_sha256"], newton_run.digest(self.exe))
                self.assertEqual(len(json.loads((self.out / "samples.json").read_text())), 2)

    def test_existing_output_requires_resume(self):
        self.invoke()
        self.assert_rejected_without_writes()

    def test_changed_binary_or_manifest(self):
        self.invoke()
        for path in (self.exe, self.manifest):
            original = path.read_bytes()
            # Whitespace changes a manifest's digest while keeping it valid JSON.
            path.write_bytes(original + b"\n")
            for options in ((), ("--resume",)):
                with self.subTest(changed=path.name, options=options):
                    self.assert_rejected_without_writes(*options)
            path.write_bytes(original)

    def test_nonempty_output_requires_saved_protocol(self):
        self.invoke()
        (self.out / "protocol.json").unlink()
        for options in ((), ("--resume",)):
            with self.subTest(options=options):
                self.assert_rejected_without_writes(*options)

    def test_malformed_saved_protocol(self):
        self.invoke()
        (self.out / "protocol.json").write_text("{broken")
        self.assert_rejected_without_writes("--resume", error=json.JSONDecodeError)

    def test_matching_resume_preserves_protocol_and_reuses_samples(self):
        self.invoke()
        protocol_path = self.out / "protocol.json"
        protocol_path.write_text(json.dumps(json.loads(protocol_path.read_text()),
                                            sort_keys=True))
        before = self.snapshot()
        self.process.reset_mock()
        self.invoke("--resume")
        after = self.snapshot()
        self.assertEqual(after.keys(), before.keys())
        for name, saved in before.items():
            self.assertEqual(after[name][0], saved[0])
            if name not in ("samples.json", "summary.json"):
                self.assertEqual(after[name], saved)
        self.process.assert_not_called()

    def test_sample_identity_rejection_preserves_protocol(self):
        self.invoke()
        protocol_path = self.out / "protocol.json"
        protocol_path.write_text(json.dumps(json.loads(protocol_path.read_text()),
                                            sort_keys=True))
        for options in (("--gpu", "1"), ("--timeout", "181")):
            with self.subTest(options=options):
                self.assert_rejected_without_writes("--resume", *options,
                                                    error=ValueError)


if __name__ == "__main__":
    unittest.main()
