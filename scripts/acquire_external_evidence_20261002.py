"""Acquire official URFD RGB+timestamps and MCFD archives, with bounded disk use."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'data/source_archives/external_evidence_20261002'
RESERVE = 48 * 1024**3
MAX_TOTAL = 10 * 1024**3


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temp.replace(path)


def transfer(row):
    url = row['url']
    group = 'URFD' if 'fenix.ur.edu.pl' in url else 'MCFD'
    path = DEST / group / url.rsplit('/', 1)[1]
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = int(row['bytes']) if row.get('bytes') else None
    if path.exists():
        if expected is not None and path.stat().st_size != expected:
            raise RuntimeError('existing file size mismatch: '+str(path))
    else:
        part = path.with_suffix(path.suffix+'.part')
        if part.exists():
            raise RuntimeError('partial file exists; inspect before resuming: '+str(part))
        if shutil.disk_usage(ROOT).free < RESERVE + (expected or 2**20):
            raise RuntimeError('48 GiB disk reserve')
        with urllib.request.urlopen(url, timeout=90) as response, part.open('xb') as out:
            while True:
                block = response.read(1024**2)
                if not block: break
                if shutil.disk_usage(ROOT).free < RESERVE:
                    raise RuntimeError('48 GiB disk reserve while downloading')
                out.write(block)
        if expected is not None and part.stat().st_size != expected:
            raise RuntimeError('download size mismatch')
        part.replace(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''): digest.update(block)
    members = None
    if path.suffix == '.zip':
        with zipfile.ZipFile(path) as z:
            assert z.testzip() is None
            members = [i.filename for i in z.infolist() if not i.is_dir()]
    result = dict(url=url, path=str(path.relative_to(ROOT)), bytes=path.stat().st_size,
                  sha256=digest.hexdigest(), zip_members=members, status='verified')
    save(path.with_suffix(path.suffix+'.manifest.json'), result)
    print(group, path.name, path.stat().st_size, 'verified', flush=True)
    return result


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    remote = json.loads((ROOT/'docs/internal/2026-10-02_external_remote_inventory.json').read_text())
    rows = [r for r in remote if 'fenix.ur.edu.pl' in r['url'] or r['url'].endswith('/dataset.zip')]
    assert len(rows) == 141
    assert all(r.get('status') == 200 and r.get('bytes') for r in rows)
    assert sum(int(r['bytes']) for r in rows) < MAX_TOTAL
    assert shutil.disk_usage(ROOT).free > RESERVE+sum(int(r['bytes']) for r in rows)
    save(DEST/'acquisition_contract.json', dict(date='2026-10-02', rows=rows,
         reserve_gib=48, budget_gib=10, rationale='separate acquisition budget; does not modify older 64GiB experiment contracts',
         source='official pages', no_training=True, no_existing_data_deleted=True))
    # Start the larger MCFD archive alongside URFD files.
    rows = sorted(rows, key=lambda r: ('fenix.ur.edu.pl' in r['url'], r['url']))
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for result in pool.map(transfer, rows):
            results.append(result)
            save(DEST/'progress.json',dict(verified=len(results),total=len(rows)))
    save(DEST/'manifest.json',dict(passed=True,files=results))


if __name__ == '__main__': main()
