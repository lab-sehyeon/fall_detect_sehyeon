"""Pinned compact global source and full-sequence causal caches for G0/G1/G2."""
import gc
import os
from pathlib import Path
import shutil
import numpy as np

from data_gen.inspect_safer_activities import load_pinned_pickle, PINNED_FILES, normalized_name
from fall_pipeline.common import global_reconstruction as math
from fall_pipeline.safer import v2_controls_reconstruction as io

ROOT=io.ROOT
CONFIG=ROOT/'configs/global_motion_document_reconstruction_v1.json'
CODE=['data_gen/global_motion_reconstruction.py','fall_pipeline/common/global_reconstruction.py',
      'data_gen/inspect_safer_activities.py']


def safety(config):
    for key in ('data_output','output_dir'):
        io.pause(ROOT/config[key])
    io.disk_gate(config)


def initialize(config):
    root=io.guard(ROOT/config['data_output']);root.mkdir(parents=True,exist_ok=True)
    contract=dict(config_sha256=io.sha256_file(CONFIG),code_sha256={p:io.sha256_file(ROOT/p) for p in CODE},
                  source_pins=PINNED_FILES,source_config_sha256=io.sha256_file(ROOT/config['source_config']))
    path=root/'contract.json'
    if path.exists():io.require(io.read(path)==contract,'global data contract changed')
    else:io.save_json(path,contract)
    for file,digest in ((ROOT/config['source_audit'],config['source_audit_sha256']),
                         (ROOT/config['sequence_root']/'manifest.json',config['sequence_manifest_sha256'])):
        io.require(io.sha256_file(file)==digest,'global parent source pin changed')
    return root,contract


def raw_assets(config,publish=lambda *a,**k:None):
    root,contract=initialize(config);source=io.read(ROOT/config['source_config'])
    manifest_path=root/'raw_manifest.json'
    journal=io.read(manifest_path) if manifest_path.exists() else {}
    for domain in ('normal','ood'):
        data=load_pinned_pickle(ROOT/source['source'][domain],domain)
        io.require(len(data['annotations'])==PINNED_FILES[domain]['sequences'],'global source population')
        for index,a in enumerate(data['annotations']):
            safety(config);uid=f'{domain}_{index:04d}'
            x,m=math.raw_channels(a,config['raw']['keypoint_confidence_min'])
            dest=root/'raw'/f'{uid}.npz';dest.parent.mkdir(exist_ok=True)
            if uid not in journal:
                with dest.with_suffix('.tmp').open('wb') as f:np.savez_compressed(f,channels=x,valid=m)
                os.replace(dest.with_suffix('.tmp'),dest)
                journal[uid]=dict(file=str(dest.relative_to(ROOT)),sha256=io.sha256_file(dest),
                                  frames=len(x),source_name=normalized_name(a['frame_dir']),
                                  width=int(a['width']),height=int(a['height']),domain=domain,
                                  valid_values=int(m.sum()),source_rebuild_exact=True)
                io.save_json(manifest_path,journal)
            io.require(io.sha256_file(dest)==journal[uid]['sha256'],'global raw changed')
            with np.load(dest,allow_pickle=False) as saved:
                io.require(set(saved.files)=={'channels','valid'},'label/prediction field in raw')
                np.testing.assert_array_equal(saved['channels'],x)
                np.testing.assert_array_equal(saved['valid'],m)
            if len(journal)%25==0:publish('raw',sequences=len(journal),total=497)
        del data,a,x,m;gc.collect()
    io.require(len(journal)==497 and sum(r['frames'] for r in journal.values())==8091357,'raw coverage')
    report=dict(passed=True,sequences=497,frames=8091357,source_rebuild_exact=True,
                labels_or_predictions_used=False,raw_manifest_sha256=io.sha256_file(manifest_path),contract=contract)
    io.save_json(root/'raw_audit.json',report);publish('raw_completed',sequences=497,total=497)
    return journal


def metadata(config,split):
    root=ROOT/config['sequence_root'];manifest=io.read(root/'manifest.json');spec=manifest['splits'][split]
    dest=root/split
    io.require(io.sha256_file(dest/'sequences.json')==spec['sequences_sha256'],'sequence order changed')
    io.require(io.sha256_file(dest/'sample_names.json')==spec['sample_names_sha256'],'sample names changed')
    arrays={}
    for name in ('sequence_index.npy','window_start.npy'):
        p=dest/name;io.require(io.sha256_file(p)==spec['payload'][name]['sha256'],'window order pin')
        arrays[name]=np.load(p,allow_pickle=False)
        io.require(arrays[name].shape==(config['split_counts'][split],),'window count')
    return io.read(dest/'sequences.json'),io.read(dest/'sample_names.json'),arrays


def cache(config,split,publish=lambda *a,**k:None,locked=False):
    io.require(split in ('train','val') or locked,'holdout features require selection lock')
    root,contract=initialize(config);journal=io.read(root/'raw_manifest.json');count=config['split_counts'][split]
    audit=io.read(root/'raw_audit.json')
    io.require(audit['passed'] and audit['contract']==contract and
               audit['raw_manifest_sha256']==io.sha256_file(root/'raw_manifest.json'),'raw audit/manifest changed')
    rows,names,meta=metadata(config,split);dest=root/split;dest.mkdir(exist_ok=True)
    path=dest/'global131.npy';complete=dest/'manifest.json'
    if complete.exists():
        report=io.read(complete)
        io.require(report['contract']==contract and report['sha256']==io.sha256_file(path),'global cache changed')
        return report
    partial=dest/'global131.partial.npy'
    array=np.lib.format.open_memmap(partial,mode='w+',dtype=np.float32,shape=(count,131))
    independent_max=0.;all_rebuild_max=0.;filled=0
    for number,row in enumerate(rows,1):
        safety(config);uid=row['uid'];record=journal[uid]
        io.require(record['source_name']==row['source_name'] and record['frames']==row['total_frames'],
                   'source/window name identity')
        lo,hi=row['output_start'],row['output_stop'];si=meta['sequence_index.npy'][lo:hi];starts=meta['window_start.npy'][lo:hi]
        io.require(lo==filled and hi-lo==row['written_window_count'],'window coverage/order')
        np.testing.assert_array_equal(si,np.full(hi-lo,row['sequence_index']))
        np.testing.assert_array_equal(starts,np.arange(0,record['frames']-63,8))
        io.require(names[lo:hi]==[f'{row["source_name"]}__f{int(s):07d}' for s in starts],'sample order differs')
        raw_path=ROOT/record['file'];io.require(io.sha256_file(raw_path)==record['sha256'],'raw source changed')
        with np.load(raw_path,allow_pickle=False) as raw:x,m=raw['channels'],raw['valid']
        causal=math.causal_arrays(x,m)
        for begin in range(0,len(starts),512):
            chunk=starts[begin:begin+512]
            array[lo+begin:lo+begin+len(chunk)]=math.features_from_causal(causal,m,chunk)
        array.flush()
        # Whole stored-cache rebuild, plus a separate scalar statistics implementation.
        for begin in range(0,len(starts),512):
            chunk=starts[begin:begin+512];expected=math.features_from_causal(causal,m,chunk)
            stored=array[lo+begin:lo+begin+len(chunk)]
            np.testing.assert_array_equal(stored,expected)
            all_rebuild_max=max(all_rebuild_max,float(np.abs(stored-expected).max(initial=0)))
        for index in sorted(set([0,len(starts)//2,len(starts)-1])) if len(starts) else []:
            expected=math.reference_feature(causal,m,int(starts[index]));actual=array[lo+index]
            np.testing.assert_allclose(actual,expected,atol=1e-6,rtol=1e-6)
            independent_max=max(independent_max,float(np.abs(actual-expected).max()))
        # Perturb only future source values; past levels/derivatives/masks must be exact.
        if len(x)>65:
            cut=len(x)//2;future=x.copy();future[cut:]+=1000;fm=m.copy();fm[cut:]=~fm[cut:]
            other=math.causal_arrays(future,fm)
            for a,b in zip(causal,other):np.testing.assert_array_equal(a[:cut],b[:cut])
        filled=hi
        if number%25==0 or number==len(rows):publish('cache',split=split,sequences=number,total=len(rows))
    io.require(filled==count and np.isfinite(array).all(),'feature population/finite')
    array.flush();del array;os.replace(partial,path)
    report=dict(passed=True,count=count,shape=[count,131],sha256=io.sha256_file(path),
                bytes=path.stat().st_size,source_manifest_sha256=io.sha256_file(root/'raw_manifest.json'),
                all_window_rebuild_max=all_rebuild_max,independent_statistics_max=independent_max,
                source_order_exact=True,full_sequence_causality_exact=True,contract=contract)
    io.save_json(complete,report);return report
