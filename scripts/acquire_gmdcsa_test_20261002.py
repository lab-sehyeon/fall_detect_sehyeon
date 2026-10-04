"""Acquire only the official OmniFall test videos at a pinned author revision."""
from pathlib import Path
import csv,hashlib,json,urllib.request,urllib.parse,concurrent.futures,shutil
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'data/source_archives/OmniFall/gmdcsa24_cs_20261002_r1'
REPO='ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos'

def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'FoundSkelModel-research-audit/1.0'}),timeout=60) as r:return r.read()
def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)
def main():
    contract=DEST/'video_contract.json'
    if not contract.exists():
        meta=json.loads(fetch(f'https://api.github.com/repos/{REPO}'))
        commit=json.loads(fetch(f'https://api.github.com/repos/{REPO}/commits/{meta["default_branch"]}'))['sha']
        tree=json.loads(fetch(f'https://api.github.com/repos/{REPO}/git/trees/{commit}?recursive=1'));assert not tree.get('truncated')
        entries={x['path']:x for x in tree['tree'] if x['type']=='blob'}
        rows=list(csv.DictReader((DEST/'metadata/splits/cs/gmdcsa24/test.csv').open()));files=[]
        for row in rows:
            path=row['path'];original=path.replace('Subject_','Subject ',1)+'.mp4';entry=entries[original]
            files.append(dict(path=path,source_path=original,git_blob=entry['sha'],bytes=entry['size'],
                url=f'https://raw.githubusercontent.com/{REPO}/{commit}/'+urllib.parse.quote(original,safe='/')))
        assert len(files)==37 and sum(x['bytes'] for x in files)<1024**3
        dump(DEST/'github_tree.json',tree)
        report=dict(repository=REPO,commit=commit,license=meta.get('license'),files=files,
            total_bytes=sum(x['bytes'] for x in files),reserve_gib=64,scope='official CS test videos only; no re-encoding')
        dump(contract,report)
        for name in ['README.md','LICENSE']:
            if name in entries:(DEST/('video_source_'+name)).write_bytes(fetch(f'https://raw.githubusercontent.com/{REPO}/{commit}/{name}'))
    spec=json.loads(contract.read_text());print('pinned',spec['commit'],'bytes',spec['total_bytes'],flush=True)
    assert shutil.disk_usage(ROOT).free>(spec['reserve_gib']*1024**3+spec['total_bytes'])
    def acquire(row):
        path=DEST/'GMDCSA24/video'/(row['path']+'.mp4');path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            raw=fetch(row['url']);assert len(raw)==row['bytes']
            assert hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()==row['git_blob']
            temp=path.with_suffix('.part');temp.write_bytes(raw);temp.replace(path)
        raw=path.read_bytes();assert len(raw)==row['bytes']
        assert hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()==row['git_blob']
        print('verified',row['path'],flush=True)
        return dict(**row,local=str(path.relative_to(ROOT)),sha256=hashlib.sha256(raw).hexdigest())
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:files=list(pool.map(acquire,spec['files']))
    dump(DEST/'video_manifest.json',dict(passed=True,files=files,repository=REPO,commit=spec['commit'],total_bytes=sum(x['bytes'] for x in files)))
    print('completed',len(files),flush=True)

if __name__=='__main__':main()
