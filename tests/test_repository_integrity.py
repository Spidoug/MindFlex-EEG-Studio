"""Public repository invariants: release metadata, docs and translations."""
from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

from mindflex import __version__


ROOT = Path(__file__).resolve().parent.parent


class RepositoryIntegrityTests(unittest.TestCase):
    def test_runtime_version_matches_package_metadata(self):
        metadata = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        version = re.search(r'^version\s*=\s*"([^"]+)"\s*$', metadata, re.MULTILINE)
        self.assertIsNotNone(version, "Project version missing in pyproject.toml")
        self.assertEqual(__version__, version.group(1))

    def test_mit_license_is_complete_and_declared(self):
        metadata = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
        self.assertIn('license = "MIT"', metadata)
        self.assertIn('Copyright (c) 2026 Douglas Santana', license_text)
        self.assertIn('THE SOFTWARE IS PROVIDED "AS IS"', license_text)

    def test_all_translations_have_same_message_keys(self):
        folder = ROOT / "mindflex" / "locales"
        english = json.loads((folder / "en.json").read_text(encoding="utf-8"))
        self.assertTrue(english)
        for catalog in folder.glob("*.json"):
            with self.subTest(catalog=catalog.name):
                content = json.loads(catalog.read_text(encoding="utf-8"))
                self.assertEqual(set(content), set(english))
                self.assertTrue(all(isinstance(value, str) for value in content.values()))

    def test_cross_platform_launcher_line_endings(self):
        attributes = (ROOT / '.gitattributes').read_text(encoding='utf-8')
        self.assertIn('*.sh text eol=lf', attributes)
        self.assertIn('*.bat text eol=crlf', attributes)
        self.assertNotIn(b'\r', (ROOT / 'RUN_STUDIO.sh').read_bytes())

    def test_markdown_local_links_resolve(self):
        markdown_files = list(ROOT.glob("*.md")) + list((ROOT / "docs").glob("*.md"))
        self.assertTrue(markdown_files)
        link_pattern = re.compile(r'!?\[[^]]*\]\(([^)]+)\)')
        for page in markdown_files:
            for match in link_pattern.finditer(page.read_text(encoding="utf-8")):
                target = match.group(1).split("#", 1)[0]
                if not target or target.startswith(("https://", "http://", "mailto:")):
                    continue
                with self.subTest(page=page.name, target=target):
                    self.assertTrue((page.parent / target).exists())


if __name__ == "__main__":
    unittest.main()
