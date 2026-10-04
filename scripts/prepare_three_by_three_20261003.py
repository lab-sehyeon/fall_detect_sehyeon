"""Official CS test videos: EDF/OCCU bounded ZIP reads; UP-Fall timestamp PNGs."""
from pathlib import Path, PurePosixPath
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse, csv, hashlib, html, io, json, os, re, shutil, sys, time, types, urllib.request, zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fall_pipeline.external import rgb_document_io as common
DEST=ROOT/'data/source_archives/ThreeByThree_20261003'
EVIDENCE=ROOT/'docs/internal/2026-10-03_three_by_three_sources'
META=ROOT/'data/source_archives/OmniFall/candidate_metadata_20261002_r1'
PACKAGE=ROOT/'data/source_archives/OmniFall/gmdcsa24_cs_20261002_r1/package_source/src'
RUNTIME=PACKAGE.parents[1]/'runtime'
sys.path[:0]=[str(PACKAGE),str(RUNTIME)]
os.environ['OMNIFALL_FFMPEG']=str(ROOT/'data/source_archives/OmniFall/le2i_cs_20261002_r1/ffmpeg_gpl')
# The official conversion module is standalone; avoid unrelated Hub dataset imports.
package=types.ModuleType('omnifall');package.__path__=[str(PACKAGE/'omnifall')];sys.modules['omnifall']=package
from omnifall import _prepare as official

def safety():
    assert shutil.disk_usage(ROOT).free>64*1024**3,'disk reserve'
    assert not (DEST/'PAUSE_REQUESTED').exists(),'pause requested'

def status(dataset,stage,**kw):
    value=dict(dataset=dataset,stage=stage,**kw)
    common.save(DEST/(dataset+'_status.json'),value);print(json.dumps(value),flush=True)

def paths(dataset):
    return sorted(r['path'] for r in csv.DictReader((META/f'splits/cs/{dataset}/test.csv').open()))

class RemoteZip(io.RawIOBase):
    """Exact bounded Range reader with four 16MiB cache blocks, no full archive."""
    def __init__(self,url,size):
        self.url=url;self.size=size;self.position=0;self.cache=OrderedDict();self.downloaded=0
        self.disk=DEST/'range_cache'/str(size);self.disk.mkdir(parents=True,exist_ok=True)
    def fetch(self,index):
        safety();block=16*1024**2;start=index*block;end=min(self.size,start+block)-1;target=self.disk/f'{index}.bin'
        if target.exists():assert target.stat().st_size==end-start+1;return target
        req=urllib.request.Request(self.url,headers={'Range':f'bytes={start}-{end}'})
        with urllib.request.urlopen(req,timeout=180) as response:
            assert response.status==206 and response.headers['Content-Range']==f'bytes {start}-{end}/{self.size}'
            data=response.read(end-start+2)
        assert len(data)==end-start+1
        partial=target.with_suffix('.part');partial.write_bytes(data);os.replace(partial,target);self.downloaded+=len(data)
        return target
    def prefetch(self,entries,dataset):
        block=16*1024**2;indices=set()
        for r in entries:
            # Local header + filename/extra lengths; conservative margin stays bounded.
            indices.update(range(r.header_offset//block,min(self.size-1,r.header_offset+r.compress_size+65536)//block+1))
        with ThreadPoolExecutor(max_workers=6) as pool:
            jobs=[pool.submit(self.fetch,index) for index in sorted(indices)]
            for i,f in enumerate(as_completed(jobs),1):
                f.result()
                if i%4==0 or i==len(jobs):status(dataset,'range_download',blocks=i,total_blocks=len(jobs),transfer_bytes=self.downloaded)
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        self.position=offset+(0 if whence==0 else self.position if whence==1 else self.size)
        assert 0<=self.position<=self.size
        return self.position
    def read(self,size=-1):
        size=self.size-self.position if size<0 else min(size,self.size-self.position)
        assert size<64*1024**2,'unbounded metadata read'
        chunks=[];block=16*1024**2
        while size:
            safety();index=self.position//block;offset=self.position%block
            if index not in self.cache:
                self.cache[index]=self.fetch(index).read_bytes()
                if len(self.cache)>4:self.cache.popitem(last=False)
            self.cache.move_to_end(index);data=self.cache[index];take=min(size,len(data)-offset)
            chunks.append(data[offset:offset+take]);self.position+=take;size-=take
        return b''.join(chunks)

def verify_video(dataset,path,video,source_frames):
    stream=official._probe(video)
    assert stream.frames==source_frames,(stream,source_frames)
    return dict(dataset=dataset,path=path,video=str(video.relative_to(ROOT)),video_sha256=common.sha(video),source_frames=source_frames)

def edf_occu(dataset):
    spec=next(r for r in common.read(EVIDENCE/'zenodo.json')['files'] if r['key']==dataset.upper()+'.zip')
    remote=RemoteZip(spec['links']['self'],spec['size']);wanted=paths(dataset);records=[]
    raw=DEST/'raw'/dataset;inventory=[]
    with zipfile.ZipFile(remote) as archive:
        entries=archive.infolist()
        common.save(DEST/(dataset+'_zip_inventory.json'),[dict(name=r.filename,size=r.file_size,compressed=r.compress_size,crc=r.CRC,offset=r.header_offset) for r in entries])
        for number,path in enumerate(wanted,1):
            safety();subject,view,stream,_=path.split('/');video=DEST/'videos'/dataset/(path+'.mp4');receipt=video.with_suffix('.json')
            if receipt.exists():
                row=common.read(receipt);assert common.sha(video)==row['video_sha256'];records.append(row);continue
            selected=[r for r in entries if '/'+subject+'/'+view+'/rgb/' in '/'+r.filename and r.filename.endswith('.bin')]
            assert selected,'no official RGB entries for '+path
            # ZIP members are shuffled; download in physical archive order.
            # The official encoder independently sorts numeric frame names.
            selected.sort(key=lambda r:r.header_offset)
            remote.prefetch(selected,dataset)
            assert len({PurePosixPath(r.filename).name for r in selected})==len(selected)
            source=raw/subject/view/'rgb';source.mkdir(parents=True,exist_ok=True)
            status(dataset,'extracting',video=number,total=len(wanted),frames=len(selected),transfer_bytes=remote.downloaded)
            for i,member in enumerate(selected):
                safety();target=source/PurePosixPath(member.filename).name
                if not target.exists():
                    data=archive.read(member);assert len(data)==member.file_size
                    target.write_bytes(data)
                else:
                    import zlib
                    assert target.stat().st_size==member.file_size and zlib.crc32(target.read_bytes())==member.CRC
                if (i+1)%500==0:status(dataset,'extracting',video=number,frame=i+1,frames=len(selected),transfer_bytes=remote.downloaded)
            video.parent.mkdir(parents=True,exist_ok=True)
            official._bin_encoder(dataset,path,source,video)()
            row=verify_video(dataset,path,video,len(selected));row['original_rgb_crc_checked']=True
            common.save(receipt,row);records.append(row)
        common.save(DEST/(dataset+'_manifest.json'),dict(passed=True,records=records,archive=spec,transfer_bytes=remote.downloaded,
            conversion='official omnifall0.2.0 _bin_encoder;30fps,libx264 CRF22,yuv420p',license='CC BY 4.0'))
    status(dataset,'completed',videos=len(records),frames=sum(r['source_frames'] for r in records))

def upfall():
    dataset='up_fall';page=html.unescape(html.unescape((EVIDENCE/'har_up.html').read_text()))
    import bisect
    trials=[(m.start(),*m.groups()) for m in official._UP_FALL_TRIAL.finditer(page)];starts=[r[0] for r in trials];links={}
    for match in official._UP_FALL_LINK.finditer(page):
        i=bisect.bisect_right(starts,match.start())-1
        if i<0:continue
        _,s,a,t=trials[i];path=f'Subject{s}/Activity{a}/Trial{t}/Subject{s}Activity{a}Trial{t}{match.group(2)}'
        assert path not in links or links[path]==match.group(1);links[path]=match.group(1)
    all_paths={r['path'] for r in csv.DictReader((META/'labels/up_fall.csv').open())}
    assert set(links)==all_paths and len(links)==1118
    common.save(EVIDENCE/'up_fall_public_camera_ids.json',links)
    wanted=paths(dataset);assert len(wanted)==264;records=[]
    for number,path in enumerate(wanted,1):
        safety();video=DEST/'videos'/dataset/(path+'.mp4');receipt=video.with_suffix('.json')
        if receipt.exists():
            row=common.read(receipt);assert common.sha(video)==row['video_sha256'];records.append(row);continue
        folder=DEST/'raw'/dataset/path;folder.mkdir(parents=True,exist_ok=True)
        name=PurePosixPath(path).name;archive_path=folder.parent/(name+'.zip')
        status(dataset,'downloading',video=number,total=264,path=path)
        if not archive_path.exists():
            url=official._drive_direct_url(links[path],progress=False)
            partial=archive_path.with_suffix('.part')
            with urllib.request.urlopen(url,timeout=180) as response,partial.open('wb') as dest:
                while True:
                    safety();block=response.read(1024**2)
                    if not block:break
                    dest.write(block)
            assert zipfile.is_zipfile(partial);os.replace(partial,archive_path)
        archive_hash=common.sha(archive_path);archive_size=archive_path.stat().st_size
        with zipfile.ZipFile(archive_path) as archive:
            members=[r for r in archive.infolist() if r.filename.lower().endswith('.png') and not r.filename.startswith('__MACOSX/')]
            assert members and len({PurePosixPath(r.filename).name for r in members})==len(members)
            for member in members:
                target=folder/PurePosixPath(member.filename).name
                data=archive.read(member);target.write_bytes(data)
        video.parent.mkdir(parents=True,exist_ok=True)
        official._up_fall_encoder(path,folder,video)()
        ordered=sorted(folder.glob('*.png'));times=[official._up_fall_time(p) for p in ordered]
        row=verify_video(dataset,path,video,len(members));row.update(archive_sha256=archive_hash,archive_bytes=archive_size,public_file_id=links[path],original_png_crc_checked=True,
            frame_timestamps=[p.stem for p in ordered],source_relative_seconds=[(t-times[0]).total_seconds() for t in times])
        common.save(receipt,row);records.append(row)
        # Only this script's temporary raw PNGs/archive, after verified encoded video.
        for member in members:(folder/PurePosixPath(member.filename).name).unlink()
        archive_path.unlink()
        status(dataset,'video_completed',video=number,total=264,frames=len(members))
    common.save(DEST/'up_fall_manifest.json',dict(passed=True,records=records,
        conversion='official omnifall0.2.0 timestamp-preserving _up_fall_encoder',license='public research download; authors citation request; no redistribution grant inferred'))
    status(dataset,'completed',videos=len(records),frames=sum(r['source_frames'] for r in records))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=['edf','occu','up_fall']);a=p.parse_args()
    if a.dataset=='up_fall':upfall()
    else:edf_occu(a.dataset)
