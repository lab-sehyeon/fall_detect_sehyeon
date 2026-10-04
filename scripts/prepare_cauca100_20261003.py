"""Acquire all 100 official CAUCA v4 AVIs; preserve the prior 19-video recipe."""
from pathlib import Path, PurePosixPath
import argparse, csv, fcntl, re, shutil, sys, zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from fall_pipeline.external import rgb_document_io as io
import prepare_omnifall_cauca_20261002 as source

DEST=ROOT/'data/source_archives/OmniFall/cauca100_20261003_r1'
OLD=ROOT/'data/source_archives/OmniFall/cauca_cs_20261002_r1'


def safety():
    assert shutil.disk_usage(ROOT).free>=64*1024**3,'disk reserve'
    assert not (DEST/'PAUSE_REQUESTED').exists(),'pause requested'
    assert sum(p.stat().st_size for p in DEST.rglob('*') if p.is_file())<4*1024**3,'source budget'


def configure():
    source.DEST=DEST;source.safety=safety;source.configure_transport()


def copy_checked(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if not dst.exists():shutil.copy2(src,dst)
    assert io.sha(src)==io.sha(dst),'copy mismatch'


def acquire():
    configure();safety()
    old_source=io.read(OLD/'source_manifest.json')
    for relative,spec in old_source['files'].items():
        assert io.sha(OLD/relative)==spec['sha256'],'prior source changed'
        copy_checked(OLD/relative,DEST/relative)
    labels=list(csv.DictReader((DEST/'metadata/labels/caucafall.csv').open()))
    paths=sorted({r['path'] for r in labels});wanted={PurePosixPath(p).name:p for p in paths}
    assert len(paths)==len(wanted)==100 and sum(r['label']=='1' for r in labels)==50
    assert len({r['path'] for r in labels if r['label']=='1'})==50
    complete=DEST/'original_manifest.json'
    if complete.exists():
        report=io.read(complete);assert report['passed'] and {r['path'] for r in report['files']}==set(paths)
        for row in report['files']:
            p=ROOT/row['local'];assert p.stat().st_size==row['bytes'] and io.sha(p)==row['sha256'] and source.crc(p)==row['official_member_crc32']
        return report['files']
    with zipfile.ZipFile(source.RemoteZip()) as archive:
        index={};all_avis=[]
        for info in archive.infolist():
            name=PurePosixPath(info.filename.replace('\\','/'))
            if name.suffix.lower()!='.avi':continue
            all_avis.append(info.filename)
            assert not name.is_absolute() and '..' not in name.parts and ':' not in str(name)
            assert name.stem in wanted and name.stem not in index,'unmapped/duplicate official video'
            subject=re.search(r'S(\d+)$',name.stem)
            assert subject and 1<=int(subject.group(1))<=10 and f'Subject.{subject.group(1)}' in name.parts
            index[name.stem]=info
        assert len(all_avis)==100 and set(index)==set(wanted),'official100 inventory differs'
    members={wanted[s]:dict(name=v.filename,bytes=v.file_size,compressed_bytes=v.compress_size,crc32=v.CRC,header_offset=v.header_offset) for s,v in index.items()}
    io.save(DEST/'archive_selected_members.json',members)
    previous={r['path']:r for r in io.read(OLD/'original_manifest.json')['files']}
    videos={r['path']:r for r in io.read(OLD/'video_manifest.json')['files']}
    rows=[]
    for number,path in enumerate(paths,1):
        safety();info=index[PurePosixPath(path).name];local=DEST/'original'/(path+'.avi')
        if path in previous:
            old=previous[path];assert old['official_member_crc32']==info.CRC and old['bytes']==info.file_size
            assert io.sha(ROOT/old['local'])==old['sha256'];copy_checked(ROOT/old['local'],local)
            v=videos[path];assert io.sha(ROOT/v['local'])==v['sha256']
            copy_checked(ROOT/v['local'],DEST/'video'/(path+'.mp4'))
            copy_checked((ROOT/v['local']).with_suffix('.json'),DEST/'video'/(path+'.json'))
        elif not local.exists():source.transport.member_fetch(info,local)
        assert local.stat().st_size==info.file_size and source.crc(local)==info.CRC
        rows.append(dict(path=path,member=info.filename,local=str(local.relative_to(ROOT)),bytes=info.file_size,
                         sha256=io.sha(local),official_member_crc32=info.CRC,reused_original=path in previous,
                         transport='official Mendeley v4 bounded Range; public original bytes, CRC verified'))
        io.save(DEST/'acquisition_progress.json',dict(stage='acquire',completed=number,total=100,files=rows))
        print('acquired',number,100,path,flush=True)
    io.save(complete,dict(passed=True,files=rows,official_archive_url=source.ZIP_URL,archive_bytes=source.ZIP_SIZE,
                         acquired_member_bytes=sum(r['bytes'] for r in rows),full_official_avi_inventory=True))
    metadata_files={str(p.relative_to(DEST)):io.sha(p) for p in sorted((DEST/'metadata').rglob('*')) if p.is_file()}
    io.save(DEST/'source_manifest.json',dict(passed=True,hf_revision=source.REV,helper_revision=source.HELPER_REV,
              official_url=source.ZIP_URL,official_archive_bytes=source.ZIP_SIZE,license='CC BY 4.0',
              files=metadata_files,scope='all100 original AVIs, all100 temporal labels; duplicate PNG frames not model inputs',
              previous19_originals_verified_reused=True,target_training=False,target_tuning=False))
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['acquire','convert','all']);args=p.parse_args()
    DEST.mkdir(parents=True,exist_ok=True)
    with (DEST/'prepare.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        rows=acquire()
        if args.stage in ['convert','all']:source.convert(rows)
    print('completed',args.stage,flush=True)


if __name__=='__main__':main()
