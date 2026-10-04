import io
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts import acquire_le2i_recovery_data as acquisition


def archive(names, symlink=False):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as stream:
        for name in names:
            info = zipfile.ZipInfo(name)
            if symlink:
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
            stream.writestr(info, b"RIFF0000AVI ")
    buffer.seek(0)
    return buffer


class Le2iAcquisitionTests(unittest.TestCase):
    def test_historical_selection(self):
        names = acquisition.selected_names()
        self.assertEqual(len(names), 261)
        self.assertEqual(sum(name.endswith(".avi") for name in names), 130)
        self.assertIn("Home_02/Home_02/Annotation_files/video (31).txt", names)
        self.assertIn("Coffee_room_02/Coffee_room_02/Annotations_files/video (50).txt", names)
        self.assertFalse(any("Office" in name or "Lecture" in name for name in names))

    def test_url_encodes_nested_path(self):
        url = acquisition.download_url("Coffee_room_01/Coffee_room_01/Videos/video (1).avi")
        self.assertIn("Coffee_room_01%2FCoffee_room_01%2FVideos%2Fvideo%20%281%29.avi", url)
        self.assertTrue(url.endswith("?datasetVersionNumber=2"))

    def test_destination_keeps_scene(self):
        self.assertEqual(acquisition.destination_name("Home_01/Home_01/Videos/video (1).avi"),
                         "source/Home_01/Videos/video (1).avi")
        with self.assertRaises(ValueError):
            acquisition.destination_name("Home_01/Home_02/Videos/video (1).avi")

    def test_unsafe_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("../escape", "/absolute", "a/../../b", "a\\b"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    acquisition.safe_path(root, name)
            (root / "link").symlink_to(root, target_is_directory=True)
            with self.assertRaises(ValueError):
                acquisition.safe_path(root, "link/file")

    def test_header(self):
        for text, expected in (("48\n80\n1,1,0,0,0,0", (48, 80)),
                               ("0\n0\n", (0, 0)), ("1,1,0,0,0,0\n2,1,0,0,0,0\n", None)):
            with self.subTest(text=text):
                self.assertEqual(acquisition.annotation_header(text), expected)

    def test_invalid_header(self):
        for text in ("0\n20\n", "30\n20\n", "-1\n20\n"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                acquisition.annotation_header(text)

    def test_zip_payload(self):
        with zipfile.ZipFile(archive(["nested/video (1).avi"])) as stream:
            self.assertEqual(acquisition.zip_payload(stream, "video (1).avi").file_size, 12)

    def test_zip_rejects_unsafe_or_multiple(self):
        for names in (["../video (1).avi"], ["/video (1).avi"], ["wrong.avi"],
                      ["video (1).avi", "other.avi"]):
            with self.subTest(names=names), zipfile.ZipFile(archive(names)) as stream:
                with self.assertRaises(ValueError):
                    acquisition.zip_payload(stream, "video (1).avi")
        with zipfile.ZipFile(archive(["video (1).avi"], symlink=True)) as stream:
            with self.assertRaises(ValueError):
                acquisition.zip_payload(stream, "video (1).avi")

    def test_raw_avi_magic_not_zip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.avi"
            path.write_bytes(b"RIFF0000AVI ")
            acquisition.validate_payload(path, "test.avi")
            path.write_bytes(b"PK0000000000")
            with self.assertRaises(ValueError):
                acquisition.validate_payload(path, "test.avi")

    def test_download_unwrap_resume_and_tamper(self):
        name = "Home_01/Home_01/Videos/video (1).avi"
        row = {"name": name, "reported_bytes": 12}
        response = archive(["video (1).avi"])
        response.headers = {"Content-Length": str(len(response.getvalue()))}
        with tempfile.TemporaryDirectory() as directory, patch.object(acquisition, "DEST", Path(directory)), patch.object(acquisition, "check_space"), patch.object(acquisition, "open_public", return_value=response) as request:
            receipt = acquisition.transfer(row)
            self.assertTrue(receipt["zip_wrapped"])
            self.assertEqual(receipt["bytes"], 12)
            self.assertEqual(acquisition.transfer(row), receipt)
            self.assertEqual(request.call_count, 1)
            target = Path(directory) / receipt["relative_path"]
            target.write_bytes(b"corrupted")
            with self.assertRaises(RuntimeError):
                acquisition.transfer(row)
            self.assertEqual(target.read_bytes(), b"corrupted")
            self.assertFalse(list(Path(directory).glob(".le2i-*")))

    def test_existing_unverified_payload_not_overwritten(self):
        name = "Home_01/Home_01/Videos/video (1).avi"
        with tempfile.TemporaryDirectory() as directory, patch.object(acquisition, "DEST", Path(directory)):
            target = Path(directory) / acquisition.destination_name(name)
            target.parent.mkdir(parents=True)
            target.write_bytes(b"user data")
            with self.assertRaises(RuntimeError):
                acquisition.transfer({"name": name, "reported_bytes": 9})
            self.assertEqual(target.read_bytes(), b"user data")


if __name__ == "__main__":
    unittest.main()
