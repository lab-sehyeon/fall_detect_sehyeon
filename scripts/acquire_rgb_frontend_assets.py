"""Download only the fixed RGB frontend releases; never load downloaded code."""
import hashlib
import json
from pathlib import Path
import os
import shutil
import requests

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'checkpoint/rgb_frontend_20260928_r1'
SPECS={
 'yolov8x.pt':dict(url='https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x.pt',
                 size=136890692,sha256='3df4ada6b4dad6d657868f2fdf7faecfb34dcfccf3a25c4b82079064718524c8'),
 'vitpose-b-multi-coco.pth':dict(url='https://huggingface.co/public-data/ViTPose/resolve/e44e7d9f97bc6f65df4e8a57d39bc02404c0aeb7/models/vitpose-b-multi-coco.pth',
                 size=360038314,sha256='595b5e128b2fe6e4364d8fa9cb9a2fd0f9710301b08ea0daa5f2903595bbab27')}

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()

def main():
    DEST.mkdir(exist_ok=True)
    for name,spec in SPECS.items():
        path=DEST/name
        if path.exists():
            assert path.stat().st_size==spec['size'] and sha(path)==spec['sha256'],'existing asset mismatch'
            continue
        assert shutil.disk_usage(ROOT).free>64*1024**3+spec['size'],'disk reserve'
        partial=path.with_suffix(path.suffix+'.partial')
        assert not partial.exists(),'partial exists: preserve and inspect before retry'
        with requests.get(spec['url'],stream=True,timeout=(20,60)) as response:
            response.raise_for_status()
            count=0
            with partial.open('xb') as output:
                for block in response.iter_content(1024**2):
                    assert not (DEST/'PAUSE_REQUESTED').exists(),'pause requested'
                    assert shutil.disk_usage(ROOT).free>64*1024**3,'disk reserve'
                    count+=len(block);assert count<=spec['size'],'oversized asset'
                    output.write(block)
            assert count==spec['size'] and sha(partial)==spec['sha256'],'asset integrity'
        os.replace(partial,path)
        print('verified',name,count,spec['sha256'],flush=True)
    report={'passed':True,'files':SPECS,
       'vitpose_distribution_basis':'ViTAE-Transformer/ViTPose README -> Gradio-Blocks/ViTPose official-linked demo -> public-data/ViTPose',
       'demo_commit':'a066142a3c5cb29fdc1fe6f4efe3f8efdd4a5aae','historical_vitpose_byte_identity_proved':False}
    path=DEST/'manifest.json'
    if path.exists():assert json.loads(path.read_text())==report
    else:path.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
