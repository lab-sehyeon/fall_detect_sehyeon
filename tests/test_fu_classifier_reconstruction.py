import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from fall_pipeline.benchmarks import fu_classifier_reconstruction as run


class ClassifierTests(unittest.TestCase):
    def fixture(self):
        c=run.io.read(run.CONFIG);rng=np.random.default_rng(11);x=rng.normal(size=(100,16)).astype('f');y=(x[:,0]>0).astype(np.int64)
        fu={'x':x,'y':y,'actions':np.where(y,5,4),'subjects':np.arange(100)%5+1,'folds':np.arange(100)%5}
        for name in c['grids']: c['grids'][name]={key:[values[0]] for key,values in c['grids'][name].items()}
        c['grids']['random_forest']['n_estimators']=[8]
        return c,fu

    def test_complete_nested_refit_and_metadata_audit(self):
        c,fu=self.fixture()
        with tempfile.TemporaryDirectory() as folder,patch.object(run,'publish'),patch.object(run.io,'disk_gate'):
            root=Path(folder);run.train(root,fu,c,{},False);report=run.audit(root,fu,c,False)
            self.assertTrue(report['passed']);self.assertTrue(report['deterministic_refit_exact'])
            for item in report['results'].values(): self.assertEqual(item['count'],100)

    def test_outer_data_cannot_change_inner_model_selection(self):
        c,fu=self.fixture();other=copy.deepcopy(fu);_,_,outer=run.core.split_indices(fu['folds'],fu['subjects'],0,True)
        other['x'][outer]*=100;other['y'][outer]=1-other['y'][outer]
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b,patch.object(run,'publish'),patch.object(run.io,'disk_gate'):
            run.train(Path(a),fu,c,{},True);run.train(Path(b),other,c,{},True)
            for name in c['grids']:
                first=run.io.read(Path(a)/name/'fold0/selection.json');second=run.io.read(Path(b)/name/'fold0/selection.json')
                self.assertEqual(first,second)

    def test_corrupt_checkpoint_rejected_before_unpickle(self):
        c,fu=self.fixture();c['grids']={'logistic':c['grids']['logistic']}
        with tempfile.TemporaryDirectory() as folder,patch.object(run,'publish'),patch.object(run.io,'disk_gate'):
            root=Path(folder);run.train(root,fu,c,{},True);path=root/'logistic/fold0/candidate_000.joblib'
            with path.open('ab') as stream: stream.write(b'test corruption')
            with self.assertRaisesRegex(RuntimeError,'candidate resume lineage'): run.train(root,fu,c,{},True)

    def test_default_predict_and_score_are_preserved(self):
        c,fu=self.fixture()
        for name,grid in c['grids'].items():
            model=run.make_model(name,list(run.ParameterGrid(grid))[0],c);run.fit(model,fu['x'],fu['y'])
            pred,score=run.outputs(model,fu['x']);np.testing.assert_array_equal(pred,model.predict(fu['x']))
            self.assertTrue(np.isfinite(score).all())


if __name__=='__main__': unittest.main()
