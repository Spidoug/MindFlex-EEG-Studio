"""Platform startup tests that do not install system packages or need hardware."""
from __future__ import annotations

import io
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from mindflex import launcher


ROOT = Path(__file__).resolve().parent.parent


@unittest.skipUnless(os.name == 'posix', 'POSIX shell required')
class LinuxShellTests(unittest.TestCase):
    def _run_shell(self, status: int = 0, *arguments: str) -> tuple[subprocess.CompletedProcess, str]:
        with tempfile.TemporaryDirectory(prefix='mindflex shell ') as folder:
            root = Path(folder)
            shutil.copy2(ROOT / 'RUN_STUDIO.sh', root / 'RUN_STUDIO.sh')
            # Substitutes for an installed Python; no dependencies or network needed.
            fake = root / 'python3'
            capture = root / 'arguments.txt'
            fake.write_text('#!/bin/sh\n'
                            'printf "%s\\n" "$@" >> "$MINDFLEX_CAPTURE"\n'
                            'if [ "$1" = "-c" ]; then exit 0; fi\n'
                            'exit "${MINDFLEX_FAKE_EXIT:-0}"\n', encoding='utf-8')
            fake.chmod(0o755)
            env = os.environ.copy()
            env.pop('VIRTUAL_ENV', None)
            env.update(PATH=f'{root}:{os.environ.get("PATH", "")}',
                       MINDFLEX_CAPTURE=str(capture), MINDFLEX_FAKE_EXIT=str(status))
            run = subprocess.run(['sh', str(root / 'RUN_STUDIO.sh'), *arguments],
                                 cwd='/', env=env, text=True, capture_output=True, timeout=10)
            return run, capture.read_text(encoding='utf-8') if capture.exists() else ''

    def test_uses_common_launcher_with_arguments_from_another_directory(self):
        result, args = self._run_shell(0, '--diagnose', '--no-gui')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('-m\nmindflex.launcher\n--diagnose\n--no-gui\n', args)

    def test_propagates_failure_without_swallowing_error(self):
        result, _ = self._run_shell(23)
        self.assertEqual(result.returncode, 23)
        self.assertIn('exited with code 23', result.stderr)
        self.assertIn('studio-startup.log', result.stderr)


class LinuxDependencyTests(unittest.TestCase):
    def test_missing_tk_reports_apt_package_without_sudo(self):
        output = io.StringIO()
        with patch.object(launcher.sys, 'platform', 'linux'), \
             patch.object(launcher, 'linux_package_manager', return_value=(
                 ['apt-get', 'install', '-y', 'python3-tk'], 'Debian/Ubuntu')):
            launcher.show_linux_tk_help(output)
        self.assertIn('sudo apt-get install python3-tk', output.getvalue())
        self.assertIn('python3-venv', output.getvalue())

    def test_missing_tk_noninteractive_never_executes_sudo(self):
        with patch.object(launcher, 'tkinter_available', return_value=False), \
             patch.object(launcher, 'linux_package_manager', return_value=(
                 ['apt-get', 'install', '-y', 'python3-tk'], 'Debian/Ubuntu')), \
             patch.object(launcher.sys.stdin, 'isatty', return_value=False), \
             patch.object(launcher, 'execute') as run:
            self.assertFalse(launcher.ensure_linux_tk(root=Path('.'), log=io.StringIO()))
        run.assert_not_called()

    def test_user_can_decline_privileged_system_install(self):
        with patch.object(launcher, 'tkinter_available', return_value=False), \
             patch.object(launcher, 'linux_package_manager', return_value=(
                 ['apt-get', 'install', '-y', 'python3-tk'], 'Debian/Ubuntu')), \
             patch.object(launcher.sys.stdin, 'isatty', return_value=True), \
             patch('builtins.input', return_value='n'), \
             patch.object(launcher, 'execute') as run:
            self.assertFalse(launcher.ensure_linux_tk(root=Path('.'), log=io.StringIO()))
        run.assert_not_called()

    def test_opt_in_install_then_recheck_tk(self):
        with patch.object(launcher, 'tkinter_available', side_effect=[False, True]), \
             patch.object(launcher, 'linux_package_manager', return_value=(
                 ['apt-get', 'install', '-y', 'python3-tk'], 'Debian/Ubuntu')), \
             patch.object(launcher.sys.stdin, 'isatty', return_value=True), \
             patch('builtins.input', return_value='yes'), \
             patch.object(launcher.os, 'geteuid', return_value=1000, create=True), \
             patch.object(launcher.shutil, 'which', return_value='/usr/bin/sudo'), \
             patch.object(launcher, 'execute', return_value=0) as run:
            self.assertTrue(launcher.ensure_linux_tk(root=Path('.'), log=io.StringIO()))
        self.assertEqual(run.call_args.args[0], ['sudo', 'apt-get', 'install', '-y', 'python3-tk'])

    def test_diagnostic_missing_tk_is_actionable_without_installing(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(launcher.sys, 'platform', 'linux'), \
                 patch.object(launcher, 'program_dir', return_value=Path(directory)), \
                 patch.object(launcher, 'tkinter_available', return_value=False), \
                 patch.object(launcher, 'linux_package_manager', return_value=(
                     ['apt-get', 'install', '-y', 'python3-tk'], 'Debian/Ubuntu')), \
                 patch.object(launcher, 'check_dependencies', return_value=False), \
                 patch.object(launcher, 'execute') as run:
                self.assertEqual(launcher.main(['--diagnose']), 1)
            run.assert_not_called()
            log = (Path(directory) / 'logs' / 'studio-startup.log').read_text(encoding='utf-8')
            self.assertIn('sudo apt-get install python3-tk', log)


if __name__ == '__main__':
    unittest.main()
