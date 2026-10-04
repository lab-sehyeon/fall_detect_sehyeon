"""Acquire only official CAUCA-CS test AVIs and apply the author conversion."""
from pathlib import Path, PurePosixPath
import argparse, csv, fcntl, hashlib, json, re, shutil, subprocess, sys, zipfile
from fractions import Fraction

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fall_pipeline.external import rgb_document_io as common
from scripts import prepare_omnifall_le2i_20261002 as transport

DEST = ROOT/'data/source_archives/OmniFall/cauca_cs_20261002_r1'
REV = '83572a37b9e3081df8c06a56874b1d1f2a19386c'
HELPER_REV = 'eb1443c1035ecdfbd15598dec16fae6aca71e7ea'
ZIP_URL = 'https://data.mendeley.com/public-api/zip/7w7fccy7ky/download/4'
ZIP_SIZE = 8326349365
FF_ROOT = ROOT/'data/source_archives/OmniFall/le2i_cs_20261002_r1/ffmpeg_gpl'
crc = transport.crc
put = transport.put


def safety():
    assert shutil.disk_usage(ROOT).free >= 64*1024**3, 'disk reserve'
    assert not (DEST/'PAUSE_REQUESTED').exists(), 'pause requested'
    assert sum(p.stat().st_size for p in DEST.rglob('*') if p.is_file()) < 16*1024**3, 'source budget'


def configure_transport():
    # Reuse bounded streaming/CRC routines; preserve frozen Le2i code on disk.
    transport.DEST=DEST; transport.ZIP_URL=ZIP_URL; transport.ZIP_SIZE=ZIP_SIZE
    transport.safety=safety


class RemoteZip(transport.RemoteZip):
    def __init__(self):
        super().__init__(base=0,size=ZIP_SIZE)

    def read(self,size=-1):
        if size < 0: size=self.size-self.position
        size=min(size,self.size-self.position)
        assert 0<=size<=16*1024**2, 'bounded central directory'
        if size==0:return b''
        with transport.ranged(self.position,size) as response:
            data=response.read(size+1)
        assert len(data)==size
        self.position+=size
        return data


def metadata():
    files={}
    names=['README.md','CONFIGS.md','LABELS.md','labels/caucafall.csv',
           *[f'splits/cs/caucafall/{s}.csv' for s in ['train','val','test']],
           'parquet/caucafall-cs/test-00000-of-00001.parquet']
    for name in names:
        url=f'https://huggingface.co/datasets/simplexsigil2/omnifall/resolve/{REV}/{name}'
        path=DEST/'metadata'/name
        if not path.exists():put(path,transport.small(url))
        files['metadata/'+name]=dict(url=url,bytes=path.stat().st_size,sha256=common.sha(path))
    url=f'https://raw.githubusercontent.com/simplexsigil/omnifall-helper-scripts/{HELPER_REV}/video_conversion/caucafall.sh'
    recipe=DEST/'metadata/caucafall.sh'
    if not recipe.exists():put(recipe,transport.small(url))
    text=recipe.read_text()
    assert '-g 20' in text and '-crf 24' in text and '-profile:v baseline' in text
    files['metadata/caucafall.sh']=dict(url=url,bytes=recipe.stat().st_size,sha256=common.sha(recipe))
    splits={s:{r['path'] for r in csv.DictReader((DEST/f'metadata/splits/cs/caucafall/{s}.csv').open())}
            for s in ['train','val','test']}
    assert all(not splits[a]&splits[b] for a,b in [('train','val'),('train','test'),('val','test')])
    assert len(splits['test'])==19
    put(DEST/'source_manifest.json',(json.dumps(dict(passed=True,hf_revision=REV,helper_revision=HELPER_REV,
        files=files,original_url=ZIP_URL,original_archive_bytes=ZIP_SIZE,license='CC BY 4.0',
        scope='official test19 AVIs only; PNG frames not used'),indent=2)+'\n').encode())
    return sorted(splits['test'])


def acquire(paths):
    target=DEST/'original_manifest.json'
    if target.exists():
        report=common.read(target)
        assert report['passed'] and {r['path'] for r in report['files']}==set(paths)
        for r in report['files']:
            assert common.sha(ROOT/r['local'])==r['sha256'] and crc(ROOT/r['local'])==r['official_member_crc32']
        return report['files']
    wanted={PurePosixPath(p).name:p for p in paths}
    assert len(wanted)==19
    with zipfile.ZipFile(RemoteZip()) as archive:
        index={}
        for info in archive.infolist():
            name=PurePosixPath(info.filename.replace('\\','/'))
            if name.suffix.lower()!='.avi' or name.stem not in wanted:continue
            assert not name.is_absolute() and '..' not in name.parts and ':' not in str(name)
            assert name.stem not in index, 'ambiguous original AVI'
            subject=re.search(r'S(\d+)$',name.stem)
            assert subject and int(subject.group(1)) in [8,9]
            assert f'Subject.{subject.group(1)}' in name.parts
            index[name.stem]=info
        assert set(index)==set(wanted), 'test AVI missing'
    selection={wanted[s]:dict(name=v.filename,bytes=v.file_size,compressed_bytes=v.compress_size,
        crc32=v.CRC,header_offset=v.header_offset) for s,v in index.items()}
    put(DEST/'archive_selected_members.json',(json.dumps(selection,indent=2)+'\n').encode())
    rows=[]
    for n,path in enumerate(paths,1):
        safety();info=index[PurePosixPath(path).name];local=DEST/'original'/(path+'.avi')
        if not local.exists():transport.member_fetch(info,local)
        assert local.stat().st_size==info.file_size and crc(local)==info.CRC
        rows.append(dict(path=path,member=info.filename,local=str(local.relative_to(ROOT)),bytes=info.file_size,
            sha256=common.sha(local),official_member_crc32=info.CRC,
            transport='official Mendeley v4 bounded Range; no signed redirect URLs recorded'))
        print('acquired',n,len(paths),path,flush=True)
    report=dict(passed=True,files=rows,official_archive_url=ZIP_URL,archive_bytes=ZIP_SIZE,
                acquired_member_bytes=sum(r['bytes'] for r in rows))
    put(target,(json.dumps(report,indent=2)+'\n').encode())
    return rows


def ffmpeg():
    m=common.read(FF_ROOT/'manifest.json')
    for name,h in m['files'].items():assert common.sha(FF_ROOT/name)==h
    own=dict(reused_root=str(FF_ROOT.relative_to(ROOT)),manifest_sha256=common.sha(FF_ROOT/'manifest.json'),
             files=m['files'],no_system_install=True)
    put(DEST/'ffmpeg_gpl/manifest.json',(json.dumps(own,indent=2)+'\n').encode())
    return FF_ROOT/'ffmpeg',FF_ROOT/'ffprobe'


def convert(rows):
    enc,pro=ffmpeg();videos=[]
    args=['-c:v','libx264','-preset','veryslow','-tune','fastdecode','-profile:v','baseline',
          '-refs','1','-bf','0','-g','20','-x264opts',
          'me=umh:subme=7:merange=24:psy-rd=1.0:aq-mode=3:aq-strength=0.8:rc-lookahead=60',
          '-crf','24','-threads','2']
    for n,row in enumerate(rows,1):
        safety();source=ROOT/row['local'];out=DEST/'video'/(row['path']+'.mp4');receipt=out.with_suffix('.json')
        assert common.sha(source)==row['sha256']
        if receipt.exists():
            r=common.read(receipt);assert common.sha(out)==r['sha256'];videos.append(r);continue
        assert not out.exists(),'unreceipted output'
        out.parent.mkdir(parents=True,exist_ok=True);part=out.with_name(out.name+'.part')
        assert not part.exists(),'inspect partial conversion'
        src=transport.probe(pro,source)
        command=[str(enc),'-hide_banner','-loglevel','error','-nostdin','-n','-threads','2','-i',str(source),
                 *args,'-f','mp4',str(part)]
        subprocess.run(command,check=True)
        dst=transport.probe(pro,part)
        assert int(src['nb_read_frames'])==int(dst['nb_read_frames'])
        assert (src['width'],src['height'])==(dst['width'],dst['height'])
        assert abs(float(Fraction(src['avg_frame_rate']))-float(Fraction(dst['avg_frame_rate'])))<.001
        assert abs(float(src['duration'])-float(dst['duration']))<1/float(Fraction(src['avg_frame_rate']))
        assert dst['codec_name']=='h264' and dst['profile']=='Constrained Baseline' and dst['pix_fmt']=='yuv420p'
        part.replace(out)
        r=dict(path=row['path'],local=str(out.relative_to(ROOT)),sha256=common.sha(out),bytes=out.stat().st_size,
               source=row,source_probe=src,output_probe=dst,command=command,source_frames=int(src['nb_read_frames']),
               recipe_revision=HELPER_REV,author_binary_identity_claim=False)
        common.save(receipt,r);videos.append(r);print('converted',n,len(rows),row['path'],flush=True)
    put(DEST/'video_manifest.json',(json.dumps(dict(passed=True,files=videos,recipe_revision=HELPER_REV),indent=2)+'\n').encode())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['acquire','convert','all']);a=parser.parse_args()
    DEST.mkdir(parents=True,exist_ok=True);configure_transport()
    with (DEST/'prepare.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        paths=metadata();rows=acquire(paths)
        if a.stage in ['convert','all']:convert(rows)
    print('completed',a.stage,flush=True)


if __name__=='__main__':main()
