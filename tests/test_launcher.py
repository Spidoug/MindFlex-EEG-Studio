"""Regression tests for Windows bootstrap, diagnostic and failure behavior."""
from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mindflex import launcher


class LauncherTests(unittest.TestCase):
    def test_supported_windows_python_range(self):
        self.assertFalse(launcher.supported_windows_python((3, 10)))
        for version in ((3, 11), (3, 12), (3, 13)):
            with self.subTest(version=version):
                self.assertTrue(launcher.supported_windows_python(version))
        self.assertFalse(launcher.supported_windows_python((3, 14)))
        self.assertFalse(launcher.supported_windows_python((3, 15)))

    def test_dependency_check_uses_requested_interpreter(self):
        with patch.object(launcher, 'execute', return_value=0) as execute:
            self.assertTrue(launcher.check_dependencies(Path('selected-python'), root=Path('.'), log=io.StringIO()))
        args = execute.call_args.args[0]
        self.assertEqual(args[0], 'selected-python')
        self.assertEqual(args[1], '-c')
        self.assertIn('Tkinter', args[2])

    def test_diagnostic_mode_does_not_install_or_launch(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(launcher, 'program_dir', return_value=Path(folder)), \
                 patch.object(launcher, 'check_dependencies', return_value=True) as checks, \
                 patch.object(launcher, 'execute') as execute:
                self.assertEqual(launcher.main(['--diagnose', '--no-gui']), 0)
            self.assertFalse(execute.called)
            self.assertFalse(checks.call_args.kwargs['check_gui'])
            self.assertTrue((Path(folder) / 'logs' / launcher.LOG_NAME).is_file())

    def test_missing_dependencies_are_installed_before_launch(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            python = root / '.venv' / 'Scripts' / 'python.exe'
            with patch.object(launcher, 'program_dir', return_value=root), \
                 patch.object(launcher.sys, 'prefix', launcher.sys.base_prefix), \
                 patch.object(launcher, 'ensure_environment', return_value=python), \
                 patch.object(launcher, 'check_dependencies', side_effect=[False, True]), \
                 patch.object(launcher, 'execute', return_value=0) as execute:
                self.assertEqual(launcher.main([]), 0)
            self.assertEqual(execute.call_count, 2)
            self.assertEqual(execute.call_args_list[0].args[0][:4], [str(python), '-m', 'pip', 'install'])
            self.assertEqual(execute.call_args_list[1].args[0], [str(python), '-m', 'mindflex'])

    def test_app_failure_propagates_and_writes_readable_log(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(launcher, 'program_dir', return_value=root), \
                 patch.object(launcher.sys, 'prefix', launcher.sys.base_prefix), \
                 patch.object(launcher, 'ensure_environment', return_value=Path('project-python')), \
                 patch.object(launcher, 'check_dependencies', return_value=True), \
                 patch.object(launcher, 'execute', return_value=19):
                self.assertEqual(launcher.main([]), 19)
            log = (root / 'logs' / launcher.LOG_NAME).read_text(encoding='utf-8')
            self.assertIn('exit code 19', log)
            self.assertIn('--diagnose', log)

    def test_failed_bluetooth_setup_allows_core_if_available(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(launcher, 'program_dir', return_value=Path(folder)), \
                 patch.object(launcher.sys, 'prefix', launcher.sys.base_prefix), \
                 patch.object(launcher, 'ensure_environment', return_value=Path('project-python')), \
                 patch.object(launcher, 'check_dependencies', side_effect=[False, True, True]), \
                 patch.object(launcher, 'execute', side_effect=[1, 0]) as execute:
                self.assertEqual(launcher.main([]), 0)
            self.assertEqual(execute.call_count, 2)
            log = (Path(folder) / 'logs' / launcher.LOG_NAME).read_text(encoding='utf-8')
            self.assertIn('Bluetooth features are unavailable', log)

    def test_batch_pauses_on_nonzero_exit(self):
        text = (Path(__file__).resolve().parent.parent / 'RUN_STUDIO.bat').read_text(encoding='utf-8')
        self.assertIn('if not "%STUDIO_EXIT_CODE%"=="0"', text)
        self.assertIn('pause', text)
        self.assertIn('mindflex.launcher', text)


if __name__ == '__main__':
    unittest.main()
