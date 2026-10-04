"""Acquire author's Subject 1 and compare with embedded HFD execution records."""
import concurrent.futures, hashlib, json, re
from pathlib import Path
from urllib.parse import quote
import cv2, requests

ROOT = Path(__file__).resolve().parents[1]
E = ROOT/'docs/internal/2026-10-04_hfd_x3d_external_comparison_evidence'
D = ROOT/'data/source_archives/HFD_GMDCSA32_20261004_r1'
tree = json.loads((E/'gmd_official_tree.json').read_text())
nb = json.loads((ROOT/'third_party/original_hfd_3dcnn_20261004/HFD_3D_CNN_EA_GitHub_.ipynb').read_text())
out = ''.join(''.join(o.get('text', [])) for c in nb['cells'] for o in c.get('outputs', []))
records = re.findall(r'GMDCSA/(Fall|ADL)/cam1/(\d+\.mp4)\n\*\*\* \[Video Info\] Number of frames: (\d+) - fps: (\d+) - chunks: (\d+)', out)
assert len(records) == 32
expected = {(a,b):dict(frames=int(c),fps=int(d),chunks=int(e),order=i) for i,(a,b,c,d,e) in enumerate(records)}
items = [x for x in tree['tree'] if x['path'].startswith('Subject 1/') and x['type']=='blob']
items += [x for x in tree['tree'] if x['path'] in ('README.md','LICENSE')]
print('download bytes',sum(x['size'] for x in items),flush=True)
def fetch(x):
    rel = x['path']; dest = D/rel; dest.parent.mkdir(parents=True, exist_ok=True)
    url = 'https://raw.githubusercontent.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos/'+tree['sha']+'/'+quote(rel)
    if not dest.exists():
        r=requests.get(url, timeout=90);r.raise_for_status();tmp=dest.with_suffix(dest.suffix+'.partial');tmp.write_bytes(r.content);tmp.rename(dest)
    b=dest.read_bytes();assert len(b)==x['size']
    assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==x['sha']
    row=dict(path=str(dest.relative_to(ROOT)),url=url,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),git_blob_sha=x['sha'])
    if rel.endswith('.mp4'):
        _,label,name=rel.split('/'); exp=expected[label,name]
        cap=cv2.VideoCapture(str(dest));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS);cap.release()
        row.update(label=int(label=='Fall'),frames=n,fps=fps,chunks=n//16,notebook_expected=exp,notebook_matches=(n==exp['frames']),fps_matches=(fps==exp['fps']))
        print(label,name,n,exp['frames'],row['notebook_matches'],flush=True)
    return row
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: rows=list(pool.map(fetch,items))
result=dict(commit=tree['sha'],files=rows,videos=sum('label' in x for x in rows),all_notebook_frame_counts_match=all(x.get('notebook_matches',True) for x in rows))
(E/'hfd_source_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
print('all counts match:',result['all_notebook_frame_counts_match'],flush=True)
