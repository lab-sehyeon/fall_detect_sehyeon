"""Read-only compact V3 sequence windows; same order/layout as reconstructed V2."""
from pathlib import Path
from collections import OrderedDict
import numpy as np
from fall_pipeline.safer import v2_controls_reconstruction as io


class SequenceWindows:
    def __init__(self,root,rows,sequence_ids,starts):
        self.root=Path(root);self.rows=rows;self.ids=sequence_ids;self.starts=starts;self.cache=OrderedDict()
        self.shape=(len(starts),3,64,25,2);self.dtype=np.dtype('float32')

    def pose(self,uid):
        if uid not in self.cache:
            self.cache[uid]=np.load(self.root/uid/'ntu25.npy',mmap_mode='r',allow_pickle=False)
            while len(self.cache)>2: self.cache.popitem(last=False)
        else: self.cache.move_to_end(uid)
        return self.cache[uid]

    def __getitem__(self,index):
        if not isinstance(index,slice) or index.step not in (None,1): raise ValueError('contiguous slice required')
        begin,end,step=index.indices(len(self.starts));ids=self.ids[begin:end];starts=self.starts[begin:end]
        result=np.zeros((len(starts),3,64,25,2),np.float32)
        for sid in np.unique(ids):
            selected=np.flatnonzero(ids==sid);pose=self.pose(self.rows[int(sid)]['uid'])
            frames=starts[selected,None]+np.arange(64)
            io.require((frames>=0).all() and (frames<len(pose)).all(),'runtime index outside sequence')
            result[selected,:,:,:,0]=pose[frames].transpose(0,3,1,2)
        return result


def verify_source(root):
    root=io.guard(root);cache=root/'sequence_cache';report=io.read(root/'final_report.json')
    io.require(report['completed'] and report['passed'] and report['research_usable'] and report['structural_integrity'] and report['model_frozen'],'V3 full gate incomplete')
    io.require(io.read(cache/'manifest.json')==report,'V3 cache/final identity')
    io.require(report['geometry_gate']['passed'] and report['policy']==io.read(root/'policy_lock.json'),'V3 policy/gate')
    io.require(io.sha256_file(root/'pilot/report.json')==report['policy']['pilot_report_sha256'],'V3 pilot report changed')
    for uid,item in report['sequence_files'].items():
        for name,spec in item['payload'].items(): io.require(io.payload(cache/uid/name)==spec,'V3 sequence payload')
        pose=np.load(cache/uid/'ntu25.npy',mmap_mode='r',allow_pickle=False)
        io.require(pose.shape==(item['frames'],25,3) and pose.dtype==np.float32 and np.all(pose[:,1]==0),'V3 sequence shape/center')
    for split,spec in report['splits'].items():
        dest=cache/split
        for name,item in spec['payload'].items(): io.require(io.payload(dest/name)==item,'V3 index payload')
        io.require(io.sha256_file(dest/'sequences.json')==spec['sequences_sha256'] and io.sha256_file(dest/'sample_names.json')==spec['sample_names_sha256'],'V3 index names/order')
        source=source_split(root,split,False);data=source['data'];count=source['count']
        # Independently reconstruct first/middle/last and both sides of every sequence boundary.
        indices={0,count//2,count-1}
        for row in data.rows:
            for i in (row['output_start']-1,row['output_start'],row['output_stop']-1):
                if 0<=i<count: indices.add(i)
        for i in sorted(indices):
            sid=int(source['arrays']['sequence_index.npy'][i]);start=int(source['arrays']['window_start.npy'][i]);uid=data.rows[sid]['uid']
            value=data[i:i+1];pose=np.load(cache/uid/'ntu25.npy',mmap_mode='r',allow_pickle=False)
            np.testing.assert_array_equal(value[0,:,:,:,0],pose[start:start+64].transpose(2,0,1))
            io.require(np.all(value[0,:,:,:,1]==0),'second person')
            labels=np.load(cache/uid/'labels.npy',mmap_mode='r',allow_pickle=False)
            io.require(source['arrays']['center_coarse_labels.npy'][i]==labels[start+32],'runtime label')
    return {'passed':True,'final_sha256':io.sha256_file(root/'final_report.json'),'sequences':report['sequences'],
            'windows':sum(s['count'] for s in report['splits'].values()),'runtime_slices_verified':True}


def source_split(root,split,smoke):
    cache=Path(root)/'sequence_cache';dest=cache/split;spec=io.read(Path(root)/'final_report.json')['splits'][split]
    count=min(spec['count'],512) if smoke else spec['count']
    arrays={name:np.load(dest/name,mmap_mode='r',allow_pickle=False)[:count] for name in io.META}
    rows=io.read(dest/'sequences.json');data=SequenceWindows(cache,rows,arrays['sequence_index.npy'],arrays['window_start.npy'])
    np.testing.assert_array_equal(arrays['center_derived_labels.npy'],np.array([0]*10+[1,2,3]+[0]*3)[arrays['center_coarse_labels.npy']])
    return {'root':dest,'count':count,'data':data,'arrays':arrays}
