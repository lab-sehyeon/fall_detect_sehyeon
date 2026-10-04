"""Verify and unpack isolated official wheels; pin the actual HF test parquet."""
from pathlib import Path
import json,hashlib,urllib.request,zipfile
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'data/source_archives/OmniFall/gmdcsa24_cs_20261002_r1'
def get(url):
    with urllib.request.urlopen(url,timeout=60) as r:return r.read()
def main():
    lock=D/'runtime_manifest.json'
    if lock.exists():
        saved=json.loads(lock.read_text())
        for name,h in saved['runtime_files'].items():assert hashlib.sha256((D/'runtime'/name).read_bytes()).hexdigest()==h
        print('verified existing isolated runtime');return
    wheels=[]
    for p in sorted((D/'runtime_wheels').glob('*.whl')):
        name,version=p.name.split('-')[:2];meta=json.loads(get(f'https://pypi.org/pypi/{name}/{version}/json'))
        source=next(r for r in meta['urls'] if r['filename']==p.name);raw=p.read_bytes();h=hashlib.sha256(raw).hexdigest()
        assert h==source['digests']['sha256'] and len(raw)==source['size']
        with zipfile.ZipFile(p) as z:
            for member in z.infolist():
                path=Path(member.filename);assert not path.is_absolute() and '..' not in path.parts
                if member.is_dir():continue
                out=D/'runtime'/path;out.parent.mkdir(parents=True,exist_ok=True);b=z.read(member)
                if out.exists():assert out.read_bytes()==b
                else:out.write_bytes(b)
        wheels.append(dict(name=name,version=version,filename=p.name,url=source['url'],sha256=h))
    assert {r['name'] for r in wheels}=={'av','pyarrow'}
    revision=json.loads((D/'source_manifest.json').read_text())['hf_revision']
    name='parquet/gmdcsa24-cs/test-00000-of-00001.parquet'
    url=f'https://huggingface.co/datasets/simplexsigil2/omnifall/resolve/{revision}/{name}'
    raw=get(url);path=D/'metadata'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    report=dict(passed=True,wheels=wheels,parquet=dict(path=str(path.relative_to(ROOT)),url=url,sha256=hashlib.sha256(raw).hexdigest()),
        runtime_files={str(p.relative_to(D/'runtime')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (D/'runtime').rglob('*') if p.is_file()},
        shared_environment_modified=False)
    lock.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='runtime_files'},indent=2))
if __name__=='__main__':main()
