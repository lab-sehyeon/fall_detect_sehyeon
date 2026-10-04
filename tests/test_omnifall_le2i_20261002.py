import io
import unittest
import zipfile
from unittest.mock import patch
from scripts import prepare_omnifall_le2i_20261002 as prep
from fall_pipeline.external.omnifall_le2i_20261002 import binary_metrics,official_rows,ASSET,OUTPUT,ROOT


class Le2iProtocolTests(unittest.TestCase):
    def test_names(self):
        for name,wanted in [
            ('Lecture room/video (4).avi','Lecture_room/video_4'),
            ('Lecture_room/Lecture room/video (15).avi','Lecture_room/video_15'),
            ('Office/Office/video (10).avi','Office/video_10'),
            ('Coffee_room_01/Videos/video (26).avi','Coffee_room_01/video_26'),
            ('Office/video_31.avi','Office/video_31')]:
            self.assertEqual(prep.canonical(name),wanted)

    def test_unsafe(self):
        for name in ['/Office/video_1.avi','../Office/video_1.avi','C:/Office/video_1.avi']:
            with self.assertRaises(AssertionError):prep.canonical(name)

    def test_nonvideo(self):self.assertIsNone(prep.canonical('Office/README.txt'))

    def test_unknown(self):
        with self.assertRaises(AssertionError):prep.canonical('Unknown/video (1).avi')

    def test_nested_offsets(self):
        inner=io.BytesIO()
        with zipfile.ZipFile(inner,'w',compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr('Office/video (1).avi',b'RIFF'+b'x'*100)
        outer=io.BytesIO()
        with zipfile.ZipFile(outer,'w',compression=zipfile.ZIP_STORED) as z:z.writestr('Office.zip',inner.getvalue())
        b=outer.getvalue()
        with patch.object(prep,'ranged',side_effect=lambda start,size:io.BytesIO(b[start:start+size])):
            with zipfile.ZipFile(prep.RemoteZip(0,len(b))) as z:
                item=z.infolist()[0];base=prep.payload_offset(item)
                with zipfile.ZipFile(prep.RemoteZip(base,item.file_size)) as n:
                    info=n.infolist()[0]
                    self.assertEqual(prep.canonical(info.filename),'Office/video_1')
                    self.assertGreater(prep.payload_offset(info,base),base)

    def test_metrics(self):
        m=binary_metrics([1,1,0,0],[1,0,1,0])
        self.assertEqual([m[k] for k in ['tp','fp','fn','tn']],[1,1,1,1])
        self.assertEqual(m['f1'],.5)

    def test_quality_reject_as_negative_retains_denominator(self):
        m=binary_metrics([1,1,0],[0,1,0])
        self.assertEqual(m['fn'],1);self.assertEqual(m['recall'],.5)
        self.assertEqual(m['f1'],2/3)

    @unittest.skipUnless((ASSET/'metadata/labels/le2i.csv').exists(),'requires acquired official metadata')
    def test_official_scope(self):
        rows,paths=official_rows()
        self.assertEqual((len(rows),len(paths),sum(r['label']==1 for r in rows)),(203,38,22))
        self.assertIn('Coffee_room_01/video_26',paths)

    @unittest.skipUnless((OUTPUT/'source_traces/Coffee_room_01_video_5.json').exists(),'requires local pixel trace')
    def test_published_le2i_eof_padding(self):
        import json
        from fall_pipeline.external.omnifall_sampling_20261002 import sampling_trace,decode
        rows,_=official_rows();r=rows[5]
        self.assertEqual(r['path'],'Coffee_room_01/video_5')
        manifest=json.loads((ASSET/'video_manifest.json').read_text())
        video=next(v for v in manifest['files'] if v['path']==r['path'])
        trace=json.loads((OUTPUT/'source_traces/Coffee_room_01_video_5.json').read_text())
        self.assertGreater(r['end'],trace['duration'])
        sample=sampling_trace(trace,r['start'],r['end'])
        pixels=decode(ROOT/video['local'],r['start'],r['end'],sample)
        self.assertEqual(pixels.shape[0],64)
        self.assertEqual(sample['unique_frames'],8)
        self.assertEqual(sample['source_indices'].count(len(trace['frames'])-1),37)


if __name__=='__main__':unittest.main()
