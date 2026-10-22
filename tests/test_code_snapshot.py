"""Portable source snapshots; synthetic files only and no stage execution."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock


class CodeSnapshotTests(unittest.TestCase):
    def test_windows_crlf_source_bytes_match_frozen_manifest_hash(self):
        from slr_pipeline import cli, common
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "slr_pipeline/synthetic.py"
            source.parent.mkdir()
            original = b'# Synthetic source, never executed.\r\nVALUE = "fixture"\r\n'
            source.write_bytes(original)
            config = {"_project_dir": str(root), "run_directory": "run",
                      "search": {}, "screening": {}, "protocol": {}}
            out = root / "run"
            with mock.patch.object(cli, "PROJECT", root):
                cli.initialize(config, out)
            snapshot = out / "code_snapshot/synthetic.py"
            self.assertEqual(snapshot.read_bytes(), original)
            manifest = common.read_json(out / "manifest.json")
            self.assertEqual(common.file_hash(snapshot), manifest["source_hashes"][str(source.relative_to(root))])


if __name__ == "__main__":
    unittest.main()
