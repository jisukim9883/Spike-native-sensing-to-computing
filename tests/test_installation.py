"""Check installation and BindsNET compatibility without changing installed packages."""

import contextlib
import io
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

import install
from install import PATCHES, apply_compatibility


class BindsNETInstallationTests(unittest.TestCase):
    def make_sources(self, root, newline="\n"):
        for relative, old, _ in PATCHES:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(("# preserved header\n" + old).replace("\n", newline).encode())

    def test_pristine_sources_are_patched_and_repeated_runs_do_not_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_sources(root)
            self.assertEqual(len(apply_compatibility(root)), 3)
            for relative, _, new in PATCHES:
                self.assertEqual((root / relative).read_bytes(), ("# preserved header\n" + new).encode())
            timestamps = {p: p.stat().st_mtime_ns for p in root.rglob("*.py")}
            self.assertEqual(apply_compatibility(root), [])
            self.assertEqual(timestamps, {p: p.stat().st_mtime_ns for p in timestamps})

    def test_crlf_sources_keep_their_line_endings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_sources(root, "\r\n")
            self.assertEqual(len(apply_compatibility(root)), 3)
            for relative, _, new in PATCHES:
                self.assertEqual((root / relative).read_bytes(),
                                 ("# preserved header\n" + new).replace("\n", "\r\n").encode())
            self.assertEqual(apply_compatibility(root), [])

    def test_unknown_source_prevents_partial_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_sources(root)
            (root / PATCHES[-1][0]).write_text("# unknown version\n")
            before = {p: p.read_bytes() for p in root.rglob("*.py")}
            with self.assertRaises(ValueError):
                apply_compatibility(root)
            self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_missing_file_prevents_partial_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_sources(root)
            (root / PATCHES[-1][0]).unlink()
            before = {p: p.read_bytes() for p in root.rglob("*.py")}
            with self.assertRaises(FileNotFoundError):
                apply_compatibility(root)
            self.assertEqual(before, {p: p.read_bytes() for p in before})


class InstallerCommandTests(unittest.TestCase):
    def test_active_python_installs_dependencies_before_patching(self):
        with patch("install.sys.version_info", (3, 11, 14)), \
                patch("install.subprocess.run") as pip, \
                patch("install.patch_installed_bindsnet") as compatibility:
            calls = Mock()
            calls.attach_mock(pip, "pip")
            calls.attach_mock(compatibility, "compatibility")
            install.main([])
            self.assertEqual(calls.mock_calls, [
                call.pip([install.sys.executable, "-m", "pip", "install", "-r",
                          str(install.ROOT / "requirements.txt")], cwd=install.ROOT, check=True),
                call.compatibility(),
            ])

    def test_patch_only_does_not_install_packages(self):
        with patch("install.sys.version_info", (3, 11, 14)), \
                patch("install.subprocess.run") as pip, \
                patch("install.patch_installed_bindsnet") as compatibility:
            install.main(["--patch-only"])
            pip.assert_not_called()
            compatibility.assert_called_once_with()

    def test_wrong_python_stops_before_installation(self):
        with patch("install.sys.version_info", (3, 10, 0)), \
                patch("install.subprocess.run") as pip, \
                patch("install.patch_installed_bindsnet") as compatibility, \
                contextlib.redirect_stderr(io.StringIO()) as errors:
            with self.assertRaises(SystemExit) as error:
                install.main([])
            self.assertEqual(error.exception.code, 2)
            self.assertIn("Python 3.11", errors.getvalue())
            pip.assert_not_called()
            compatibility.assert_not_called()

    def test_pip_failure_stops_before_patching(self):
        failure = subprocess.CalledProcessError(1, ["pip"])
        with patch("install.sys.version_info", (3, 11, 14)), \
                patch("install.subprocess.run", side_effect=failure), \
                patch("install.patch_installed_bindsnet") as compatibility:
            with self.assertRaises(subprocess.CalledProcessError):
                install.main([])
            compatibility.assert_not_called()

    def test_wrong_bindsnet_version_is_rejected(self):
        with patch("install.importlib.metadata.distribution", return_value=Mock(version="0.3.0")), \
                patch("install.apply_compatibility") as compatibility:
            with self.assertRaisesRegex(ValueError, "Expected bindsnet==0.2.7"):
                install.patch_installed_bindsnet()
            compatibility.assert_not_called()

    def test_active_distribution_supplies_patch_location(self):
        package_dir = Path("/test-environment/site-packages/bindsnet")
        distribution = Mock(version="0.2.7")
        distribution.locate_file.return_value = package_dir
        with patch("install.importlib.metadata.distribution", return_value=distribution) as lookup, \
                patch("install.apply_compatibility", return_value=[]) as compatibility:
            install.patch_installed_bindsnet()
            lookup.assert_called_once_with("bindsnet")
            distribution.locate_file.assert_called_once_with("bindsnet")
            compatibility.assert_called_once_with(package_dir)


if __name__ == "__main__":
    unittest.main()
