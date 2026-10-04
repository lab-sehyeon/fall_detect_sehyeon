"""Read-only source gates for separately reconstructed joint model lineages."""
from pathlib import Path
import numpy as np
import torch
from fall_pipeline.safer import v2_controls_reconstruction as io

ROOT=io.ROOT
META=io.META


def prepare(config,lineage):
    spec=config['lineages'][lineage];sources={};root=io.guard(ROOT/spec['data_root'])
    if lineage=='v1':
        dev=ROOT/spec['development_cache'];holdout=ROOT/spec['holdout_cache']
        io.require(io.sha256_file(dev/'feature_cache_manifest.json')==spec['cache_manifest_sha256'],'V1 feature root pin')
        io.require(io.sha256_file(holdout/'f0b_postselection_report.json')==spec['holdout_report_sha256'],'V1 holdout root pin')
        report=io.read(dev/'feature_cache_manifest.json')
        io.require(report['research_usable'] and report['integrity']['passed'] and report['model_state_hash_before']==report['model_state_hash_after'],'V1 cache integrity')
        io.require(Path(report['run_contract']['data_root'])==root,'V1 source lineage')
        for split in ('train','val'):
            path=dev/split/'feature_manifest.json'
            io.require(io.sha256_file(path)==report['splits'][split]['feature_manifest_sha256'],'V1 split pin')
            sources[split]={'path':str(dev/split),'manifest':str(path),'sha256':io.sha256_file(path)}
        # No holdout arrays or stored metrics are loaded at development setup.
        for split in ('test','ood'):
            path=holdout/split/'result.json';sources[split]={'path':str(holdout/split),'manifest':str(path),'sha256':io.sha256_file(path)}
        lineage_pin=spec['cache_manifest_sha256']
    else:
        controls=ROOT/spec['controls_root'];report=io.read(controls/'final_report.json')
        io.require(report['passed'] and report['research_usable'] and report['model_frozen'],'V2 control incomplete')
        io.require(report['contract']['config_sha256']==spec['controls_config_sha256'],'V2 controls config')
        io.require(io.sha256_file(controls/'independent_audit.json')==report['audit_sha256'] and io.read(controls/'independent_audit.json')['passed'],'V2 controls audit')
        for split in ('train','val','test','ood'):
            path=controls/'cache'/split/'manifest.json'
            sources[split]={'path':str(path.parent),'manifest':str(path),'sha256':io.sha256_file(path)}
        lineage_pin=io.sha256_file(controls/'final_report.json')
    for item in sources.values(): io.guard(item['path']);io.guard(item['manifest'])
    return {'lineage':lineage,'data_root':str(root),'lineage_pin':lineage_pin,'sources':sources,'counts':spec['split_counts']}


def load_safer(prepared,split,locked=False,limit=None):
    io.require(split in ('train','val') or locked,'holdout arrays require all-fold lock')
    source=prepared['sources'][split];dest=Path(source['path']);path=Path(source['manifest'])
    io.require(io.sha256_file(path)==source['sha256'],'source manifest changed')
    report=io.read(path);arrays={};n=prepared['counts'][split]
    names=('temporal_features.npy','spatial_features.npy')+META
    for name in names:
        p=dest/name;spec=report['payload'][name]
        io.require(p.stat().st_size==spec['bytes'] and io.sha256_file(p)==spec['sha256'],'feature payload: '+name)
        a=np.load(p,mmap_mode='r',allow_pickle=False)
        expected=(n,1024) if name.endswith('features.npy') else (n,)
        io.require(a.shape==expected and a.dtype==(np.float32 if len(expected)==2 else np.int64),'cache shape/dtype')
        io.require(all(np.isfinite(a[i:i+8192]).all() for i in range(0,len(a),8192)),'cache finite')
        arrays[name]=a[:limit] if limit else a
    for name in META:
        original=np.load(Path(prepared['data_root'])/split/name,mmap_mode='r',allow_pickle=False)
        np.testing.assert_array_equal(arrays[name],original[:limit] if limit else original)
    labels=arrays['center_derived_labels.npy'];coarse=arrays['center_coarse_labels.npy']
    np.testing.assert_array_equal(labels,np.array([0]*10+[1,2,3]+[0]*3)[coarse])
    x=np.concatenate((arrays['temporal_features.npy'],arrays['spatial_features.npy']),axis=1)
    return {'x':x,'y':np.array(labels,copy=True),'source_manifest_sha256':source['sha256']}


def load_fu(config):
    spec=config['fu'];root=io.guard(ROOT/spec['root']);path=root/'preprocess_manifest.json'
    io.require(io.sha256_file(path)==spec['manifest_sha256'],'FU preprocessing pin')
    manifest=io.read(path);io.require(manifest['integrity']['passed'],'FU gate')
    for name,wanted in manifest['output_hashes_sha256'].items(): io.require(io.sha256_file(root/name)==wanted,'FU metadata/raw changed')
    io.require(io.sha256_file(ROOT/spec['probe_report'])==spec['probe_report_sha256'] and io.read(ROOT/spec['probe_report'])['passed'],'FU probe audit')
    io.require(io.sha256_file(ROOT/spec['features'])==spec['features_sha256'],'FU features changed')
    with np.load(ROOT/spec['features'],allow_pickle=False) as saved:
        x=np.concatenate((saved['t'],saved['s']),axis=1)
    arrays={key:np.load(root/name,allow_pickle=False) for key,name in
            (('y','labels.npy'),('actions','action_ids.npy'),('folds','fold_ids.npy'),('subjects','subjects.npy'))}
    io.require(x.shape==(993,2048) and np.isfinite(x).all(),'FU feature shape')
    io.require(all(v.shape==(993,) for v in arrays.values()),'FU metadata count')
    np.testing.assert_array_equal(arrays['folds'],(arrays['subjects']-1)%5)
    np.testing.assert_array_equal(arrays['y'],(arrays['actions']==5).astype(np.int64))
    io.require(int(arrays['y'].sum())==165 and int((arrays['actions']==4).sum())==168,'FU class support')
    return {'x':x,**arrays}


def to_device(data,device):
    return {**data,'x':torch.from_numpy(data['x']).to(device),'y':torch.from_numpy(data['y']).long().to(device)}
