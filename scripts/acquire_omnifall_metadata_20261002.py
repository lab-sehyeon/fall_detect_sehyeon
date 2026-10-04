"""Pin official benchmark metadata and inspect the published companion source."""
from pathlib import Path
import hashlib,json,urllib.request,tarfile
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'data/source_archives/OmniFall/gmdcsa24_cs_20261002_r1'

def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'FoundSkelModel-research-audit/1.0'}),timeout=60) as r:return r.read()
def put(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        assert path.read_bytes()==data,'refuse changed existing source '+str(path)
    else:
        temp=path.with_suffix(path.suffix+'.part');temp.write_bytes(data);temp.replace(path)
def main():
    DEST.mkdir(parents=True,exist_ok=True);lock=DEST/'source_manifest.json'
    if lock.exists():
        report=json.loads(lock.read_text())
        for p,s in report['files'].items():assert hashlib.sha256((DEST/p).read_bytes()).hexdigest()==s['sha256']
        print(json.dumps({k:v for k,v in report.items() if k!='files'},indent=2));return
    info=json.loads(fetch('https://huggingface.co/api/datasets/simplexsigil2/omnifall'));rev=info['sha']
    put(DEST/'hf_info.json',(json.dumps(info,indent=2)+'\n').encode())
    wanted=['README.md','CONFIGS.md','STRUCTURE.md','LABELS.md','labels/GMDCSA24.csv',
            'splits/cs/gmdcsa24/test.csv','splits/cs/gmdcsa24/train.csv','splits/cs/gmdcsa24/val.csv']
    available={x['rfilename'] for x in info['siblings']}
    wanted=[x for x in wanted if x in available]+[x for x in available if x.endswith('.py')]
    files={}
    for name in sorted(wanted):
        url=f'https://huggingface.co/datasets/simplexsigil2/omnifall/resolve/{rev}/{name}'
        b=fetch(url);put(DEST/'metadata'/name,b);files['metadata/'+name]=dict(url=url,sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
    py=json.loads(fetch('https://pypi.org/pypi/omnifall/0.2.0/json'));sdist=next(x for x in py['urls'] if x['packagetype']=='sdist')
    b=fetch(sdist['url']);assert hashlib.sha256(b).hexdigest()==sdist['digests']['sha256']
    pp=DEST/sdist['filename'];put(pp,b);files[pp.name]=dict(url=sdist['url'],sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
    with tarfile.open(pp) as tar:
        for member in tar.getmembers():
            if not member.isfile():continue
            parts=Path(member.name).parts;assert len(parts)>1 and '..' not in parts and not member.name.startswith('/')
            if member.name.endswith(('.py','.md','.toml','.cfg')) or parts[-1]=='LICENSE':
                put(DEST/'package_source'/Path(*parts[1:]),tar.extractfile(member).read())
    report=dict(passed=True,time=datetime.now(timezone.utc).isoformat(),hf_revision=rev,package_version='0.2.0',files=files,
        original_video_acquired=False,scope='metadata and official companion source only')
    put(lock,(json.dumps(report,indent=2)+'\n').encode());print(json.dumps({k:v for k,v in report.items() if k!='files'},indent=2))

if __name__=='__main__':main()
