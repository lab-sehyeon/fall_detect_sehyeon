"""Public pinned Le2i test assets only; no system installs or model execution."""
from pathlib import Path, PurePosixPath
import argparse, csv, fcntl, hashlib, io, json, os, re, shutil, stat, struct
import subprocess, sys, tarfile, time, urllib.request, urllib.parse, zipfile, zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fall_pipeline.external import rgb_document_io as common

DEST = ROOT / 'data/source_archives/OmniFall/le2i_cs_20261002_r1'
OLD = ROOT / 'data/source_archives/Le2i-FDD/kaggle_v2_20261002_r1/source'
REV = '83572a37b9e3081df8c06a56874b1d1f2a19386c'
HELPER_REV = 'eb1443c1035ecdfbd15598dec16fae6aca71e7ea'
ZIP_URL = 'https://search-data.ubfc.fr/imvia/dl_data.php?file=101'
ZIP_SIZE = 9608701260
BUILD = 'autobuild-2026-10-01-13-06'
FF_NAME = 'ffmpeg-n8.1.3-14-g330caae0c1-linux64-gpl-8.1.tar.xz'


def safety():
    assert shutil.disk_usage(ROOT).free >= 64 * 1024**3, 'disk reserve'
    assert not (DEST / 'PAUSE_REQUESTED').exists(), 'pause requested'
    assert sum(p.stat().st_size for p in DEST.rglob('*') if p.is_file()) <= 16 * 1024**3, 'source budget'


def put(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert not path.is_symlink(), 'symlink target'
    if path.exists():
        assert path.read_bytes() == data, 'changed existing source'
        return
    part = path.with_name(path.name + '.part')
    with part.open('wb') as f:
        f.write(data)
    part.replace(path)


def request(url, headers=None):
    for attempt in range(3):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers={
                'User-Agent': 'FoundSkelModel-research/1.0', **(headers or {})}), timeout=60)
        except Exception as e:
            if attempt == 2:
                raise RuntimeError(f'public fetch failed: {type(e).__name__}, HTTP {getattr(e,"code",None)}') from None
            time.sleep(2)


def small(url, limit=8*1024**2):
    with request(url) as r:
        b = r.read(limit+1)
    assert len(b) <= limit
    return b


def metadata():
    names = ['README.md', 'CONFIGS.md', 'LABELS.md', 'labels/le2i.csv',
             *[f'splits/cs/le2i/{s}.csv' for s in ['train', 'val', 'test']],
             'parquet/le2i/test-00000-of-00001.parquet']
    files = {}
    for name in names:
        url = f'https://huggingface.co/datasets/simplexsigil2/omnifall/resolve/{REV}/{name}'
        path = DEST / 'metadata' / name
        if not path.exists(): put(path, small(url))
        files['metadata/'+name] = dict(url=url, sha256=common.sha(path), bytes=path.stat().st_size)
    url = f'https://raw.githubusercontent.com/simplexsigil/omnifall-helper-scripts/{HELPER_REV}/video_conversion/le2i.sh'
    path = DEST / 'metadata/le2i.sh'
    if not path.exists(): put(path, small(url))
    files['metadata/le2i.sh'] = dict(url=url, sha256=common.sha(path), bytes=path.stat().st_size)
    put(DEST/'source_manifest.json', (json.dumps(dict(passed=True, hf_revision=REV,
        helper_revision=HELPER_REV, files=files, original_url=ZIP_URL, original_archive_bytes=ZIP_SIZE,
        license='CC BY-NC-SA 3.0', scope='official test videos only'), indent=2)+'\n').encode())
    splits = {s: {r['path'] for r in csv.DictReader((DEST/f'metadata/splits/cs/le2i/{s}.csv').open())}
              for s in ['train', 'val', 'test']}
    assert all(not splits[a] & splits[b] for a,b in [('train','val'),('train','test'),('val','test')])
    assert len(splits['test']) == 38
    return sorted(splits['test'])


class RemoteZip(io.RawIOBase):
    """Only ZIP metadata is read this way; member payloads use one streamed Range."""
    def __init__(self, base=0, size=ZIP_SIZE):
        self.position = 0; self.base = base; self.size = size
        assert 0 <= base < ZIP_SIZE and 0 < size <= ZIP_SIZE-base
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.position
    def seek(self, offset, whence=0):
        assert whence in (0,1,2)
        self.position = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        assert 0 <= self.position <= self.size
        return self.position
    def read(self, size=-1):
        if size < 0: size = self.size-self.position
        size = min(size, self.size-self.position)
        assert 0 <= size <= 2*1024**2, 'unbounded archive metadata read'
        if size == 0: return b''
        start = self.base+self.position
        with ranged(start, size) as r: data = r.read(size+1)
        assert len(data) == size
        self.position += size
        return data


def ranged(start, size):
    safety()
    r = request(ZIP_URL, {'Range': f'bytes={start}-{start+size-1}'})
    if r.status != 206 or r.headers.get('Content-Range') != f'bytes {start}-{start+size-1}/{ZIP_SIZE}':
        r.close()
        raise RuntimeError('server did not honor exact bounded Range')
    return r


def canonical(name):
    p = PurePosixPath(name.replace('\\','/'))
    assert not p.is_absolute() and '..' not in p.parts and ':' not in name, 'unsafe archive path'
    if p.suffix.lower() != '.avi': return None
    scenes = {'coffeeroom01':'Coffee_room_01', 'coffeeroom02':'Coffee_room_02',
              'home01':'Home_01', 'home02':'Home_02', 'lectureroom':'Lecture_room', 'office':'Office'}
    found = [scenes[re.sub(r'[^a-z0-9]','',x.lower())] for x in p.parts[:-1]
             if re.sub(r'[^a-z0-9]','',x.lower()) in scenes]
    number = re.fullmatch(r'video[ _]*(?:\((\d+)\)|(\d+))', p.stem, flags=re.I)
    assert len(set(found)) == 1 and number, 'unresolved AVI name'
    return f'{found[0]}/video_{int(number.group(1) or number.group(2))}'


def crc(path):
    c = 0
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024**2), b''): c = zlib.crc32(b, c)
    return c & 0xffffffff


def payload_offset(info, base=0):
    with ranged(base+info.header_offset, 30) as r: header = r.read()
    assert header[:4] == b'PK\x03\x04'
    nlen, xlen = struct.unpack_from('<HH', header, 26)
    return base+info.header_offset+30+nlen+xlen


def member_fetch(info, dest, base=0):
    assert info.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
    assert not info.flag_bits & 1 and not stat.S_ISLNK(info.external_attr >> 16)
    assert 0 < info.file_size < 1024**3 and 0 < info.compress_size < 1024**3
    start = payload_offset(info, base)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name+'.part')
    dec = zlib.decompressobj(-15) if info.compress_type == zipfile.ZIP_DEFLATED else None
    size = 0; consumed = 0; began = time.monotonic()
    with ranged(start, info.compress_size) as r, part.open('wb') as f:
        while True:
            b = r.read(1024**2)
            if not b: break
            consumed += len(b)
            assert consumed <= info.compress_size
            out = dec.decompress(b, info.file_size-size+1) if dec else b
            size += len(out)
            assert size <= info.file_size and not (dec and dec.unconsumed_tail), 'oversized decompression'
            f.write(out); safety()
            delay = consumed/(16*1024**2) - (time.monotonic()-began)
            if delay > 0: time.sleep(min(delay,1))
        if dec:
            out = dec.flush(); f.write(out); size += len(out)
            assert dec.eof and not dec.unused_data
    assert consumed == info.compress_size and size == info.file_size and crc(part) == info.CRC
    assert not dest.exists(), 'do not replace completed source'
    part.replace(dest)


def mirror_fetch(info, dest, canonical_path):
    """Alternate transport only; accept solely against the official member."""
    inventory=common.read(OLD.parent/'source_inventory.json')
    names=[r['name'] for r in inventory['files']
           if r['name'].lower().endswith('.avi') and canonical(r['name'])==canonical_path]
    assert len(names)==1,'mirror mapping ambiguous'
    url='https://www.kaggle.com/api/v1/datasets/download/tuyenldvn/falldataset-imvia/'+urllib.parse.quote(names[0],safe='')+'?datasetVersionNumber=2'
    payload=DEST/'mirror_transport'/(canonical_path+'.zip')
    payload.parent.mkdir(parents=True,exist_ok=True)
    if not payload.exists():
        part=payload.with_name(payload.name+'.part'); total=0; began=time.monotonic()
        with request(url) as r, part.open('wb') as f:
            assert r.status==200
            while True:
                b=r.read(1024**2)
                if not b:break
                total+=len(b);assert total<1024**3
                f.write(b);safety()
                delay=total/(16*1024**2)-(time.monotonic()-began)
                if delay>0:time.sleep(min(delay,1))
        assert zipfile.is_zipfile(part),'unexpected mirror transport'
        part.replace(payload)
    with zipfile.ZipFile(payload) as z:
        members=[r for r in z.infolist() if not r.is_dir()]
        assert len(members)==1
        member=members[0];name=PurePosixPath(member.filename.replace('\\','/'))
        assert not name.is_absolute() and '..' not in name.parts and ':' not in str(name)
        assert not stat.S_ISLNK(member.external_attr>>16) and name.suffix.lower()=='.avi'
        assert member.file_size==info.file_size and member.CRC==info.CRC,'mirror differs from official member'
        dest.parent.mkdir(parents=True,exist_ok=True);part=dest.with_name(dest.name+'.mirror.part');size=0
        with z.open(member) as src,part.open('wb') as f:
            while True:
                b=src.read(1024**2)
                if not b:break
                size+=len(b);assert size<=info.file_size
                f.write(b);safety()
        assert size==info.file_size and crc(part)==info.CRC
        assert not dest.exists();part.replace(dest)
    return dict(kind='Kaggle v2 transport, checked against official member',url=url,
                payload=str(payload.relative_to(ROOT)),payload_sha256=common.sha(payload))


def acquire(paths, transport='official'):
    report_path = DEST/'original_manifest.json'
    if report_path.exists():
        report = common.read(report_path)
        assert {r['path'] for r in report['files']} == set(paths)
        for r in report['files']: assert common.sha(ROOT/r['local']) == r['sha256']
        return report['files']
    # Official archive contains six STORED scene ZIPs, so their central
    # directories and member payloads remain seekable without downloading all.
    with zipfile.ZipFile(RemoteZip()) as z:
        index = {}; outer_records=[]
        for outer in z.infolist():
            if not outer.filename.lower().endswith('.zip'): continue
            assert outer.compress_type == zipfile.ZIP_STORED and not outer.flag_bits & 1
            base=payload_offset(outer)
            outer_records.append(dict(name=outer.filename,base=base,bytes=outer.file_size,crc32=outer.CRC))
            with zipfile.ZipFile(RemoteZip(base,outer.file_size)) as nested:
                for info in nested.infolist():
                    name = canonical(info.filename)
                    if name in paths:
                        assert name not in index
                        index[name] = (info,base,outer.filename)
    assert set(index) == set(paths)
    common.save(DEST/'archive_selected_members.json',dict(outer=outer_records,
        selected={p:dict(name=i.filename,parent=parent,base=base,size=i.file_size,
        compressed_size=i.compress_size,crc32=i.CRC,header_offset=i.header_offset) for p,(i,base,parent) in index.items()}))
    rows = []
    for n, path in enumerate(paths,1):
        safety(); info,base,parent = index[path]; scene,stem = path.split('/'); number = int(stem.split('_')[-1])
        previous = OLD/scene/'Videos'/f'video ({number}).avi'
        reused = previous.is_file() and previous.stat().st_size == info.file_size and crc(previous) == info.CRC
        local = previous if reused else DEST/'original'/(path+'.avi')
        provenance=dict(kind='existing Kaggle v2 file' if reused else 'official Range',url=ZIP_URL)
        if not local.exists():
            if transport=='kaggle-v2':provenance=mirror_fetch(info,local,path)
            else:member_fetch(info, local, base)
        assert local.stat().st_size == info.file_size and crc(local) == info.CRC
        rows.append(dict(path=path,member=parent+'!'+info.filename,local=str(local.relative_to(ROOT)),
            sha256=common.sha(local),bytes=info.file_size,official_member_crc32=info.CRC,
            reused_existing=reused,transport=provenance,identity='official archive member size+CRC32; local SHA256 pinned'))
        print('acquired',n,len(paths),path,'reused' if reused else 'downloaded',flush=True)
    common.save(report_path,dict(passed=True,files=rows,official_archive_url=ZIP_URL,archive_bytes=ZIP_SIZE,
                                downloaded_bytes=sum(r['bytes'] for r in rows if not r['reused_existing'])))
    return rows


def ffmpeg():
    base = DEST/'ffmpeg_gpl'; manifest = base/'manifest.json'
    if manifest.exists():
        m = common.read(manifest)
        for name,h in m['files'].items(): assert common.sha(base/name) == h
        return base/'ffmpeg', base/'ffprobe'
    urlbase = f'https://github.com/BtbN/FFmpeg-Builds/releases/download/{BUILD}'
    checksums = small(urlbase+'/checksums.sha256')
    put(base/'checksums.sha256',checksums)
    matches = [s.split()[0] for s in checksums.decode().splitlines() if s.split()[-1].lstrip('*') == FF_NAME]
    assert len(matches) == 1
    archive = base/FF_NAME
    if not archive.exists():
        part = base/(FF_NAME+'.part'); total=0
        with request(urlbase+'/'+FF_NAME) as r, part.open('wb') as f:
            while True:
                b=r.read(1024**2)
                if not b:break
                total += len(b); assert total < 400*1024**2
                f.write(b); safety()
        assert common.sha(part) == matches[0]; part.replace(archive)
    assert common.sha(archive) == matches[0]
    files={}
    with tarfile.open(archive) as t:
        for member in t.getmembers():
            if Path(member.name).name not in ['ffmpeg','ffprobe','LICENSE.txt']:continue
            assert member.isfile() and not member.issym() and member.size < 400*1024**2
            name=Path(member.name).name; put(base/name,t.extractfile(member).read()); files[name]=common.sha(base/name)
            if name in ['ffmpeg','ffprobe']: (base/name).chmod(0o755)
    assert {'ffmpeg','ffprobe'} <= files.keys()
    common.save(manifest,dict(url=urlbase+'/'+FF_NAME,archive_sha256=matches[0],files=files,no_system_install=True))
    return base/'ffmpeg',base/'ffprobe'


def probe(binary, video):
    return json.loads(subprocess.check_output([str(binary),'-v','error','-select_streams','v:0',
        '-count_frames','-show_streams','-of','json',str(video)]))['streams'][0]


def convert(rows):
    enc, pro = ffmpeg(); videos=[]
    args=['-pix_fmt','yuv420p','-c:v','libx264','-preset','veryslow','-tune','fastdecode',
          '-profile:v','baseline','-refs','1','-bf','0','-g','25','-x264opts',
          'me=umh:subme=7:merange=24:psy-rd=1.0:aq-mode=3:aq-strength=0.8:rc-lookahead=60',
          '-crf','20','-threads','2']
    for n,row in enumerate(rows,1):
        safety(); source=ROOT/row['local']; out=DEST/'video'/(row['path']+'.mp4'); receipt=out.with_suffix('.json')
        assert common.sha(source)==row['sha256']
        if receipt.exists():
            r=common.read(receipt); assert common.sha(out)==r['sha256']; videos.append(r); continue
        assert not out.exists(), 'unreceipted completed video'
        out.parent.mkdir(parents=True,exist_ok=True); part=out.with_name(out.name+'.part')
        assert not part.exists(), 'inspect previous partial conversion before retry'
        src=probe(pro,source)
        cmd=[str(enc),'-hide_banner','-loglevel','error','-nostdin','-n','-threads','2','-i',str(source),
             *args,'-f','mp4',str(part)]
        subprocess.run(cmd,check=True)
        dst=probe(pro,part)
        assert int(src['nb_read_frames'])==int(dst['nb_read_frames'])
        assert (src['width'],src['height'])==(dst['width'],dst['height'])
        from fractions import Fraction
        assert abs(float(Fraction(src['avg_frame_rate']))-float(Fraction(dst['avg_frame_rate'])))<.001
        assert abs(float(src['duration'])-float(dst['duration']))<1/float(Fraction(src['avg_frame_rate']))
        assert dst['codec_name']=='h264' and dst['profile']=='Constrained Baseline' and dst['pix_fmt']=='yuv420p'
        part.replace(out)
        r=dict(path=row['path'],local=str(out.relative_to(ROOT)),sha256=common.sha(out),bytes=out.stat().st_size,
            source=row,source_probe=src,output_probe=dst,command=cmd,source_frames=int(src['nb_read_frames']),
            recipe_revision=HELPER_REV,author_binary_identity_claim=False)
        common.save(receipt,r);videos.append(r);print('converted',n,len(rows),row['path'],flush=True)
    common.save(DEST/'video_manifest.json',dict(passed=True,files=videos,recipe_revision=HELPER_REV))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['acquire','convert','all'])
    parser.add_argument('--transport',choices=['official','kaggle-v2'],default='official');a=parser.parse_args()
    DEST.mkdir(parents=True,exist_ok=True)
    with (DEST/'prepare.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        paths=metadata(); rows=acquire(paths,a.transport)
        if a.stage in ['convert','all']:convert(rows)
    print('completed',a.stage,flush=True)


if __name__=='__main__':main()
