import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import acquire_le2i_recovery_data as acquisition
from scripts import audit_le2i_acquisition as audit


class Le2iAuditTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        names = sorted(acquisition.selected_names())
        remaining = {"Coffee_room_01": 47, "Coffee_room_02": 12, "Home_01": 30, "Home_02": 7}
        rows = []
        for name in names:
            path = root / acquisition.destination_name(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            if name == "README.txt":
                body = b"test source only"
            elif name.endswith(".avi"):
                body = b"RIFF0000AVI "
            else:
                scene = name.split("/")[0]
                if scene + "/" + path.name in acquisition.EXCLUDED:
                    body = b"1,1,0,0,0,0\n2,1,0,0,0,0\n"
                elif remaining[scene] > 0:
                    body = b"1\n2\n"
                    remaining[scene] -= 1
                else:
                    body = b"0\n0\n"
            path.write_bytes(body)
            rows.append({"source_name": name, "relative_path": str(path.relative_to(root)),
                         "bytes": len(body), "reported_bytes": len(body), "transport_bytes": len(body),
                         "sha256": hashlib.sha256(body).hexdigest(), "zip_wrapped": False})
        for filename, data in {
            "manifest.json": {"passed": True, "phase": "all", "files": rows},
            "acquisition_contract.json": {"selected_source_names": names},
            "annotation_audit.json": {"passed": True},
        }.items():
            (root / filename).write_text(json.dumps(data))
        return root

    def test_complete_readback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(directory)
            with patch.object(audit, "DEST", root), contextlib.redirect_stdout(io.StringIO()):
                audit.main()
            result = json.loads((root / "download_verification.json").read_text())
            self.assertTrue(result["passed"])
            self.assertEqual(result["video_annotation_pairs"], 130)
            self.assertEqual(result["valid_header_videos"], 127)
            self.assertEqual(result["fall"], 96)
            self.assertFalse(result["full_decode_verified"])
            self.assertFalse(result["historical_byte_identity_verified"])

    def test_tampered_file_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(directory)
            (root / "source/README.txt").write_bytes(b"altered source only")
            with patch.object(audit, "DEST", root), self.assertRaises(RuntimeError):
                audit.main()
            self.assertFalse((root / "download_verification.json").exists())

    def test_extra_file_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(directory)
            (root / "source/unexpected.txt").write_text("other data")
            with patch.object(audit, "DEST", root), self.assertRaises(RuntimeError):
                audit.main()


if __name__ == "__main__":
    unittest.main()
