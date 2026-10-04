import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import test_fu_classifier_reconstruction as fixtures
from scripts import run_fu_reference_r3 as run


class ReferenceTests(unittest.TestCase):
    def test_classifiers_replay_and_corrupt_oof_rejected(self):
        config,fu=fixtures.ClassifierTests().fixture()
        with tempfile.TemporaryDirectory() as folder,patch.object(run.clf,'publish'),patch.object(run.io,'disk_gate'):
            root=Path(folder);run.clf.train(root,fu,config,{},False)
            report=run.clf.audit(root,fu,config,False)
            result,payload=run.replay_classifiers(root,fu,report)
            self.assertEqual(result,report['results']);self.assertEqual(len(payload),12)
            pred=np.load(root/'logistic/oof_pred.npy');pred[0]=1-pred[0]
            run.io.save_array(root/'logistic/oof_pred.npy',pred)
            with self.assertRaises(AssertionError): run.replay_classifiers(root,fu,report)

    def test_joint_selected_cpu_and_exact_oof_replay(self):
        c=run.io.read(run.joint.CONFIG);c['model']['feature_dim']=16
        _,fu=fixtures.ClassifierTests().fixture()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);acc={s:[] for s in ('j0','j1')}
            for k in range(5):
                dest=root/'nested'/f'fold{k}';dest.mkdir(parents=True)
                outer=np.flatnonzero(fu['folds']==k);run.io.save_array(dest/'outer_indices.npy',outer)
                run.io.save_json(dest/'selection_lock.json',{'test':k})
                record={'selection_lock_sha256':run.io.sha256_file(dest/'selection_lock.json'),'results':{}}
                initial=None
                for phase in ('j0','j1'):
                    model=run.clf.core.model_for(phase,c,torch.device('cpu'),initial);initial=model.state_dict()
                    base=dest/phase;base.mkdir();weight=base/'epoch_001.pt';torch.save(model.state_dict(),weight)
                    run.io.save_json(base/'result.json',{'best':{'epoch':1,'weight_sha256':run.io.sha256_file(weight)}})
                    pred=run.clf.core.predict(model,torch.from_numpy(fu['x'][outer]),'fu',4096)
                    path=dest/f'{phase}_outer_logits.npy';run.io.save_array(path,pred)
                    record['results'][phase]={'logits_sha256':run.io.sha256_file(path)};acc[phase].extend(zip(outer.tolist(),pred))
                run.io.save_json(dest/'outer_result.json',record)
            final={'nested':{}}
            for phase,rows in acc.items():
                rows.sort(key=lambda row:row[0]);ids=np.array([row[0] for row in rows]);logits=np.stack([row[1] for row in rows])
                run.io.save_array(root/'nested'/f'{phase}_oof_indices.npy',ids);run.io.save_array(root/'nested'/f'{phase}_oof_logits.npy',logits)
                final['nested'][phase]={'metrics':run.clf.core.binary_metrics(fu['y'],fu['actions'],logits)}
            metrics,payload,maximum=run.replay_joint(root,fu,c,final)
            self.assertEqual(maximum,0);self.assertEqual(len(payload),4)
            self.assertEqual(metrics['j1'],final['nested']['j1']['metrics'])
            run.io.save_array(root/'nested/j1_oof_indices.npy',np.zeros(len(fu['y']),np.int64))
            with self.assertRaises(AssertionError): run.replay_joint(root,fu,c,final)

    def test_queue_pause_blocks_reference(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(run,'OUTPUT',Path(folder)/'reference'),patch.object(run.queue,'QUEUE',Path(folder)):
            (Path(folder)/'PAUSE_REQUESTED').touch()
            with self.assertRaises(run.io.PauseRequested): run.check_pause()


if __name__=='__main__': unittest.main()
