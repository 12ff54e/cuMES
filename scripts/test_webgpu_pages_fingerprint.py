"""Exercise Pages dependency discovery with a real CMake/Ninja build."""

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from webgpu_pages_fingerprint import fingerprint


class PagesFingerprintTest(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[2] / "tmp"
        scratch.mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(prefix="pages-deps-", dir=scratch)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source, self.build = self.root / "source tree", self.root / "build tree"
        self.write("CMakeLists.txt", """
cmake_minimum_required(VERSION 3.20)
project(PagesDependencies LANGUAGES C)
set(PROVENANCE first CACHE STRING "Build revision")
set(BEHAVIOR 1 CACHE STRING "Compiler setting")
file(GLOB_RECURSE sources CONFIGURE_DEPENDS src/*.c)
add_executable(browser ${sources})
target_compile_definitions(browser PRIVATE
  CUMES_GIT_REVISION="${PROVENANCE}" CUMES_GIT_DIRTY=0 BEHAVIOR=${BEHAVIOR})
add_subdirectory(deps/math)
target_link_libraries(browser PRIVATE math)
add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/theme.txt
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/assets/theme.txt
          ${CMAKE_BINARY_DIR}/theme.txt
  DEPENDS ${CMAKE_SOURCE_DIR}/assets/theme.txt VERBATIM)
add_custom_target(pages DEPENDS browser ${CMAKE_BINARY_DIR}/theme.txt)
""")
        self.write("src/main.c", '#include "../headers/options.h"\nint helper(void);\nint main(void) { return VALUE + helper(); }\n')
        self.write("headers/options.h", "#define VALUE 0\n")
        self.write("deps/math/CMakeLists.txt", "add_library(math STATIC helper.c)\n")
        self.write("deps/math/helper.c", "int helper(void) { return 0; }\n")
        self.write("assets/theme.txt", "initial theme\n")
        self.write("service worker.js", "// packaging-only input\n")
        self.write("docs/README.md", "unrelated documentation\n")
        self.inputs = self.root / "package-inputs.json"
        self.inputs.write_text(json.dumps([str(self.source / "service worker.js")]))
        self.rebuild()

    def write(self, path, contents):
        path = self.source / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)

    def rebuild(self, *options):
        for command in (
            ["cmake", "-S", str(self.source), "-B", str(self.build), "-G", "Ninja", *options],
            ["cmake", "--build", str(self.build), "--target", "pages"],
        ):
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def current(self):
        return fingerprint(self.source, self.build, "pages", self.inputs)

    def test_real_dependencies_and_unrelated_changes(self):
        before, record = self.current()
        self.assertIn("headers/options.h", record["files"])
        self.assertIn("deps/math/helper.c", record["files"])
        self.assertNotIn("docs/README.md", record["files"])
        self.write("docs/README.md", "edited documentation\n")
        self.write("native/solver.cu", "// not part of this build\n")
        self.rebuild()
        self.assertEqual(before, self.current()[0])
        for path, contents in (
            ("headers/options.h", "#define VALUE 1\n"),
            ("deps/math/helper.c", "int helper(void) { return 1; }\n"),
            ("assets/theme.txt", "updated theme\n"),
            ("service worker.js", "// updated worker\n"),
        ):
            with self.subTest(path=path):
                self.write(path, contents)
                self.rebuild()
                after = self.current()[0]
                self.assertNotEqual(before, after)
                before = after

    def test_project_reorganization_and_new_glob_inputs(self):
        before = self.current()[0]
        self.write("new/location/value.h", "#define EXTRA 3\n")
        self.write("src/extra.c", '#include "../new/location/value.h"\nint extra(void) { return EXTRA; }\n')
        self.rebuild()
        after, record = self.current()
        self.assertNotEqual(before, after)
        self.assertIn("new/location/value.h", record["files"])
        (self.source / "src/extra.c").rename(self.source / "src/renamed.c")
        self.rebuild()
        renamed, record = self.current()
        self.assertNotEqual(after, renamed)
        self.assertIn("src/renamed.c", record["files"])
        self.assertNotIn("src/extra.c", record["files"])
        (self.source / "src/renamed.c").unlink()
        self.rebuild()
        self.assertEqual(before, self.current()[0])

    def test_compiler_flags_and_provenance(self):
        before = self.current()[0]
        self.rebuild("-DPROVENANCE=another-commit")
        self.assertEqual(before, self.current()[0])
        self.rebuild("-DBEHAVIOR=2")
        self.assertNotEqual(before, self.current()[0])

    def test_checkout_location_and_missing_dependency_log(self):
        before = self.current()[0]
        relocated = self.root / "relocated source"
        shutil.copytree(self.source, relocated)
        self.source, self.build = relocated, self.root / "relocated build"
        self.inputs.write_text(json.dumps([str(self.source / "service worker.js")]))
        self.rebuild()
        self.assertEqual(before, self.current()[0])
        (self.build / ".ninja_deps").unlink()
        with self.assertRaisesRegex(RuntimeError, "No compiler dependencies"):
            self.current()


if __name__ == "__main__":
    unittest.main()
