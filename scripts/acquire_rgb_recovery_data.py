"""Pinned OOPS-Fall818 streaming extraction and SAFER OOD30 acquisition only."""
import argparse
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path,PurePosixPath
import shutil
import sys
import tarfile
import time
import urllib.request
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
CONTROL=ROOT/'data/source_archives/rgb_acquisition_20260923'
OOPS=ROOT/'data/source_archives/OmniFall/oops_original_818_20260923'
SAFER=ROOT/'data/source_archives/SAFER-Activities'
URL='https://oops.cs.columbia.edu/data/video_and_anns.tar.gz'
REPOS={'oops':('simplexsigil2/omnifall','83572a37b9e3081df8c06a56874b1d1f2a19386c'),
       'safer_ood':('SAFER-Activities/SAFER-Activities','5b3131df295514a1399da1acf83d908d8324b3a9')}
RESERVE=64*1024**3
OOPS_BUDGET=6*1024**3
META=['data_files/oops_video_mapping.csv','labels/OOPS.csv','labels/label2id.csv','prepare_oops_videos.py']


def require(ok,message):
    if not ok: raise RuntimeError(message)


def read(path): return json.loads(path.read_text())


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        while block:=stream.read(1024**2): digest.update(block)
    return digest.hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');os.replace(temp,path)


def state(dataset,stage,**fields):
    value={'dataset':dataset,'stage':stage,'time':datetime.now(timezone.utc).isoformat(),**fields}
    save(CONTROL/(dataset+'_status.json'),value);print(json.dumps(value,ensure_ascii=False),flush=True)


def safe_path(root,name):
    p=PurePosixPath(name)
    require(name and not p.is_absolute() and '..' not in p.parts and '\\' not in name,'unsafe relative path')
    dest=root/str(p)
    require(dest.resolve().is_relative_to(root.resolve()) and not dest.is_symlink(),'path outside destination or symlink')
    return dest


def pause_space(extra=0):
    require(not (CONTROL/'PAUSE_REQUESTED').exists(),'pause requested')
    require(shutil.disk_usage(ROOT).free>=RESERVE+extra,'64GiB disk reserve')


def verify(path,spec):
    require(path.is_file() and path.stat().st_size==spec['size'],'download size mismatch')
    if spec.get('lfs'): require(sha(path)==spec['lfs']['sha256'],'LFS checksum mismatch')
    else:
        body=path.read_bytes();git=hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest()
        require(git==spec['blobId'],'Git blob checksum mismatch')
    return {'bytes':path.stat().st_size,'sha256':sha(path)}


def metadata(dataset):
    repo,rev=REPOS[dataset];path=CONTROL/(dataset+'_source_manifest.json')
    if path.exists():
        manifest=read(path);require((manifest['repo'],manifest['revision'])==(repo,rev),'source revision changed');return manifest
    request='https://huggingface.co/api/datasets/'+repo+'/revision/'+rev+'?blobs=true'
    with urllib.request.urlopen(request,timeout=60) as stream: data=json.load(stream)
    require(data['sha']==rev,'API revision changed')
    files=[f for f in data['siblings'] if f['rfilename'] in META] if dataset=='oops' else [f for f in data['siblings'] if f['rfilename'].startswith('raw/SAFER-Activites-Test-Dataset/Videos/')]
    files.sort(key=lambda f:f['rfilename'])
    require(len(files)==(4 if dataset=='oops' else 30),'asset count mismatch')
    if dataset=='safer_ood': require(sum(f['size'] for f in files)==8563103484,'SAFER payload bytes')
    manifest={'repo':repo,'revision':rev,'files':files};save(path,manifest);return manifest


def load_mapping(path):
    with path.open(newline='') as f: rows=list(csv.DictReader(f))
    mapping={r['oops_path']:r['itw_path'] for r in rows}
    require(len(rows)==len(mapping)==818 and len(set(mapping.values()))==818,'mapping count or duplicate')
    for source,dest in mapping.items():
        safe_path(OOPS,source);safe_path(OOPS,dest)
        require(source.startswith('oops_video/') and dest.startswith('falls/') and dest.endswith('.mp4'),'mapping schema')
    return mapping


def prepare_oops():
    manifest=metadata('oops');OOPS.mkdir(parents=True,exist_ok=True)
    for spec in manifest['files']:
        dest=safe_path(OOPS/'metadata',spec['rfilename']);dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():
            link='https://huggingface.co/datasets/'+manifest['repo']+'/resolve/'+manifest['revision']+'/'+spec['rfilename']
            with urllib.request.urlopen(link,timeout=60) as stream: body=stream.read(spec['size']+1)
            require(len(body)==spec['size'],'metadata size mismatch')
            temp=dest.with_suffix(dest.suffix+'.part');temp.write_bytes(body);verify(temp,spec);os.replace(temp,dest)
        verify(dest,spec)
    mapping=load_mapping(OOPS/'metadata/data_files/oops_video_mapping.csv')
    with (OOPS/'metadata/labels/OOPS.csv').open(newline='') as f: labels=list(csv.DictReader(f))
    require(len(labels)==4022,'historical annotation count mismatch')
    require({r['path']+'.mp4' for r in labels}==set(mapping.values()),'annotation/mapping video identity')
    save(CONTROL/'oops_metadata_audit.json',{'passed':True,'videos':818,'segments':len(labels),'source_manifest_sha256':sha(CONTROL/'oops_source_manifest.json')})
    return mapping


def video_probe(path):
    import cv2
    cv2.setNumThreads(1);cap=cv2.VideoCapture(str(path))
    try:
        require(cap.isOpened(),'video open failed')
        n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=float(cap.get(cv2.CAP_PROP_FPS));w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ok,frame=cap.read();require(ok and n>0 and fps>0 and w>0 and h>0,'video first frame or metadata invalid')
        cap.set(cv2.CAP_PROP_POS_FRAMES,max(0,n-1));last,_=cap.read();require(last,'video last frame decode failed')
        return {'frames':n,'fps':fps,'width':w,'height':h,'first_last_decode':True,'full_frame_decode':False}
    finally: cap.release()


class CountedStream:
    def __init__(self,source,callback): self.source=source;self.callback=callback;self.count=0;self.started=time.monotonic();self.last=self.started
    def read(self,n):
        require(n>0,'unbounded stream read');block=self.source.read(n);self.count+=len(block)
        elapsed=time.monotonic()-self.started;delay=self.count/(16*1024**2)-elapsed
        if delay>0: time.sleep(min(delay,1))
        if time.monotonic()-self.last>30:
            pause_space();self.callback(self.count);self.last=time.monotonic()
        return block


def extract_inner(stream,mapping,journal,root=OOPS):
    found=set();total=sum(v['bytes'] for v in journal.values())
    with tarfile.open(fileobj=stream,mode='r|gz',bufsize=1024**2) as archive:
        for member in archive:
            if member.name not in mapping: continue
            require(member.name not in found and member.isfile(),'duplicate/nonregular selected archive member');found.add(member.name)
            name=mapping[member.name];dest=safe_path(root,name);pause_space(member.size)
            require(0<member.size<=512*1024**2,'unexpected video size')
            if name in journal:
                require(dest.stat().st_size==member.size==journal[name]['bytes'] and sha(dest)==journal[name]['sha256'],'completed video changed');continue
            require(not dest.exists(),'unmanaged destination; refusing overwrite')
            require(total+member.size<=OOPS_BUDGET,'OOPS selected output budget')
            dest.parent.mkdir(parents=True,exist_ok=True);temp=dest.with_suffix('.mp4.part')
            require(not temp.is_symlink(),'partial symlink');copied=0
            with archive.extractfile(member) as source,temp.open('wb') as out:
                while block:=source.read(1024**2):
                    out.write(block);copied+=len(block)
            require(copied==member.size,'truncated selected video')
            probe=video_probe(temp);digest=sha(temp);os.replace(temp,dest)
            journal[name]={'bytes':copied,'sha256':digest,'original_member':member.name,'probe':probe};total+=copied
            save(CONTROL/'oops_videos.json',journal)
            if len(journal)%25==0: state('oops','streaming',completed=len(journal),expected=len(mapping),stored_bytes=total)
    require(found==set(mapping),'archive missing requested videos')


def oops():
    mapping=prepare_oops();path=CONTROL/'oops_videos.json';journal=read(path) if path.exists() else {}
    for name,record in journal.items():
        require(name in mapping.values() and sha(safe_path(OOPS,name))==record['sha256'],'resume manifest mismatch')
    if len(journal)<len(mapping) or not (CONTROL/'oops_transfer_audit.json').exists():
        pause_space(OOPS_BUDGET-sum(r['bytes'] for r in journal.values()))
        req=urllib.request.Request(URL,headers={'User-Agent':'FoundSkelModel-recovery/1.0'})
        with urllib.request.urlopen(req,timeout=120) as response:
            require(response.status==200 and int(response.headers['Content-Length'])==47904996151,'OOPS archive size changed')
            remote={'url':URL,'bytes':47904996151,'etag':response.headers.get('ETag'),'last_modified':response.headers.get('Last-Modified')}
            remote_path=CONTROL/'oops_remote.json'
            if remote_path.exists(): require(read(remote_path)==remote,'archive origin changed')
            else: save(remote_path,remote)
            counter=CountedStream(response,lambda count:state('oops','streaming',network_bytes=count,completed=len(journal),expected=818))
            seen=False
            with tarfile.open(fileobj=counter,mode='r|gz',bufsize=1024**2) as outer:
                for member in outer:
                    if member.name=='oops_dataset/video.tar.gz':
                        require(not seen and member.isfile(),'outer video archive invalid');seen=True
                        with outer.extractfile(member) as inner: extract_inner(inner,mapping,journal)
            while counter.read(1024**2): pass
            require(seen and counter.count==remote['bytes'],'incomplete archive transfer')
            save(CONTROL/'oops_transfer_audit.json',{'passed':True,'compressed_bytes':counter.count,'archive_stored':False,'remote':remote})
    require(len(journal)==818 and (CONTROL/'oops_transfer_audit.json').exists(),'incomplete OOPS evidence')
    report={'passed':True,'videos':818,'segments':4022,'bytes':sum(r['bytes'] for r in journal.values()),'videos_manifest_sha256':sha(path),'source_manifest_sha256':sha(CONTROL/'oops_source_manifest.json'),'transcoded':False,'official_per_video_checksums_available':False,'full_frame_decode':False}
    save(CONTROL/'oops_final_report.json',report);state('oops','completed',**report)


def safer():
    from huggingface_hub import hf_hub_download
    manifest=metadata('safer_ood');journal={};remaining=sum(f['size'] for f in manifest['files']);pause_space(remaining+OOPS_BUDGET)
    for spec in manifest['files']:
        pause_space(spec['size']);dest=safe_path(SAFER,spec['rfilename'])
        if not dest.exists():
            hf_hub_download(repo_id=manifest['repo'],filename=spec['rfilename'],repo_type='dataset',revision=manifest['revision'],local_dir=str(SAFER),cache_dir=str(CONTROL/'hf_cache'),token=True,etag_timeout=60)
        result=verify(dest,spec);result['probe']=video_probe(dest);journal[spec['rfilename']]=result
        save(CONTROL/'safer_ood_videos.json',journal);state('safer_ood','downloading',completed=len(journal),expected=30)
    report={'passed':True,'videos':len(journal),'bytes':sum(r['bytes'] for r in journal.values()),'source_manifest_sha256':sha(CONTROL/'safer_ood_source_manifest.json'),'videos_manifest_sha256':sha(CONTROL/'safer_ood_videos.json'),'all_lfs_sha256_match':True,'transcoded':False,'full_frame_decode':False}
    save(CONTROL/'safer_ood_final_report.json',report);state('safer_ood','completed',**report)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--dataset',choices=('oops','safer_ood'),required=True);parser.add_argument('--metadata-only',action='store_true');args=parser.parse_args()
    require(Path(sys.prefix).name=='fall_detect' and os.environ.get('CUDA_VISIBLE_DEVICES')=='','fall_detect CPU-only required')
    require(all(p.resolve().is_relative_to(ROOT) for p in (CONTROL,OOPS,SAFER)),'output outside project')
    CONTROL.mkdir(parents=True,exist_ok=True)
    with (CONTROL/(args.dataset+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        contract={'code_sha256':sha(Path(__file__)),'repo':REPOS[args.dataset],'reserve_bytes':RESERVE,'oops_budget_bytes':OOPS_BUDGET}
        contract['repo']=list(contract['repo']);path=CONTROL/(args.dataset+'_contract.json')
        if path.exists(): require(read(path)==contract,'download code/contract changed')
        else: save(path,contract)
        try:
            pause_space();state(args.dataset,'preflight')
            if args.metadata_only:
                prepare_oops() if args.dataset=='oops' else metadata('safer_ood');state(args.dataset,'metadata_ready');return
            oops() if args.dataset=='oops' else safer()
        except Exception as error:
            # Never print HTTP signed URLs, authentication headers or token-bearing exceptions.
            state(args.dataset,'paused' if (CONTROL/'PAUSE_REQUESTED').exists() else 'failed',error_type=type(error).__name__,detail=str(error) if type(error) is RuntimeError else 'Transfer/validation failed; completed files preserved.')
            raise SystemExit(1)


if __name__=='__main__': main()
