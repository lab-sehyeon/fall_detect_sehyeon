import hashlib
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
from scripts import acquire_rgb_recovery_data as run


class AcquisitionTests(unittest.TestCase):
    def archive(self,names):
        stream=io.BytesIO()
        with tarfile.open(fileobj=stream,mode='w:gz') as archive:
            for name,kind in names:
                member=tarfile.TarInfo(name);member.type=kind;member.size=4 if kind==tarfile.REGTYPE else 0
                archive.addfile(member,io.BytesIO(b'data') if member.size else None)
        stream.seek(0);return stream

    def test_path_traversal_absolute_and_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'out';root.mkdir()
            for name in ('../escape','/tmp/escape','a/../../escape','a\\escape'):
                with self.assertRaises(RuntimeError): run.safe_path(root,name)
            (root/'link').symlink_to(Path(folder),target_is_directory=True)
            with self.assertRaises(RuntimeError): run.safe_path(root,'link/file')

    def test_selective_extract_resume_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(run,'pause_space'),patch.object(run,'video_probe',return_value={'test':True}):
            root=Path(folder);mapping={'oops_video/train/one.mp4':'falls/one.mp4'};journal={}
            with patch.object(run,'CONTROL',root/'control'):
                for _ in range(2): run.extract_inner(self.archive([('unused',tarfile.REGTYPE),('oops_video/train/one.mp4',tarfile.REGTYPE)]),mapping,journal,root/'data')
                self.assertEqual(len(journal),1);self.assertEqual((root/'data/falls/one.mp4').read_bytes(),b'data')
                self.assertFalse((root/'data/unused').exists())
                with self.assertRaisesRegex(RuntimeError,'unmanaged'): run.extract_inner(self.archive([('oops_video/train/one.mp4',tarfile.REGTYPE)]),mapping,{},root/'data')

    def test_link_duplicate_and_missing_rejected(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(run,'pause_space'),patch.object(run,'video_probe',return_value={}):
            root=Path(folder);mapping={'x':'falls/x.mp4'}
            with patch.object(run,'CONTROL',root/'control'):
                with self.assertRaisesRegex(RuntimeError,'nonregular'): run.extract_inner(self.archive([('x',tarfile.SYMTYPE)]),mapping,{},root)
                with self.assertRaisesRegex(RuntimeError,'missing'): run.extract_inner(self.archive([('other',tarfile.REGTYPE)]),mapping,{},root)
                with self.assertRaisesRegex(RuntimeError,'duplicate'): run.extract_inner(self.archive([('x',tarfile.REGTYPE),('x',tarfile.REGTYPE)]),mapping,{},root)

    def test_lfs_git_blob_and_size(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'payload';path.write_bytes(b'data')
            run.verify(path,{'size':4,'lfs':{'sha256':hashlib.sha256(b'data').hexdigest()}})
            run.verify(path,{'size':4,'blobId':hashlib.sha1(b'blob 4\0data').hexdigest()})
            with self.assertRaisesRegex(RuntimeError,'checksum'): run.verify(path,{'size':4,'lfs':{'sha256':'0'*64}})
            with self.assertRaisesRegex(RuntimeError,'size'): run.verify(path,{'size':5})

    def test_pause_disk_budget(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(run,'CONTROL',Path(folder)):
            with patch.object(run.shutil,'disk_usage',return_value=type('Usage',(),{'free':run.RESERVE-1})()):
                with self.assertRaisesRegex(RuntimeError,'reserve'): run.pause_space()
            (Path(folder)/'PAUSE_REQUESTED').touch()
            with self.assertRaisesRegex(RuntimeError,'pause'): run.pause_space()


if __name__=='__main__': unittest.main()
