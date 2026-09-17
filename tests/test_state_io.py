#!/usr/bin/env python3
"""Read actual C++ typed-input fixtures through the plotting readers.

Usage: python3 tests/test_state_io.py /path/to/tests/test_asymmetric_io
"""

from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cumes_plot.state_io import ASYM_FAM_NAMES, FAM_NAMES, load_state


class StateIoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="cumes_state_io_")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        result = subprocess.run([str(FIXTURE_WRITER), str(cls.directory)],
                                capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)

    def test_all_containers_preserve_large_and_empty_inputs(self):
        for name in ("large", "empty"):
            for suffix in (".bin", ".ckpt", ".nc", ".h5"):
                path = self.directory / (name + suffix)
                if not path.exists():  # optional C++ output backend
                    continue
                with self.subTest(name=name, suffix=suffix):
                    ns, mnmax, families, params, _ = load_state(path)
                    self.assertEqual((ns, mnmax), (5, 6))
                    self.assertEqual(tuple(families), FAM_NAMES + ASYM_FAM_NAMES)
                    for index, values in enumerate(families.values()):
                        np.testing.assert_array_equal(
                            values, index * 100.0 + np.arange(30) + 0.125)
                    self.assertTrue(params["lasym"])
                    self.assertEqual(params["raxis_s"], [0.0, 0.002])
                    self.assertEqual(params["zaxis_c"], [0.03, -0.001])
                    rbs = [] if name == "empty" else [
                        (1, 1, 0.01), (2, -1, 0.002), (1, 1, -0.003),
                    ] + [(2, -1, 0.0)] * 10000
                    zbc = [] if name == "empty" else [
                        (0, 0, 0.04), (1, -1, 0.03), (0, 1, 0.001),
                    ] + [(1, 1, 0.0)] * 10000
                    self.assertEqual(params["rbs"], rbs)
                    self.assertEqual(params["zbc"], zbc)

    def test_json_record_versions_are_rejected(self):
        for suffix, version in ((".bin", 9), (".ckpt", 7)):
            path = self.directory / ("old" + suffix)
            payload = bytearray((self.directory / ("large" + suffix)).read_bytes())
            struct.pack_into("<i", payload, 8, version)
            path.write_bytes(payload)
            with self.subTest(suffix=suffix), self.assertRaisesRegex(
                    SystemExit, "JSON-based asymmetric input record"):
                load_state(path)

    def test_partial_hdf5_boundary_is_rejected(self):
        source = self.directory / "empty.h5"
        if not source.exists():
            self.skipTest("HDF5 output backend unavailable")
        import h5py
        path = self.directory / "partial.h5"
        shutil.copyfile(source, path)
        with h5py.File(path, "r+") as output:
            del output["zbc_n"]
        with self.assertRaisesRegex(ValueError, "incomplete asymmetric boundary"):
            load_state(path)

    def test_truncated_input_is_rejected(self):
        source = self.directory / "large.ckpt"
        path = self.directory / "truncated.ckpt"
        path.write_bytes(source.read_bytes()[:-8])
        with self.assertRaises((ValueError, struct.error, SystemExit)):
            load_state(path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    FIXTURE_WRITER = Path(sys.argv.pop(1)).resolve()
    unittest.main()
