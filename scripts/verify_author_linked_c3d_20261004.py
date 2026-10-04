"""Load exact HFD author-defined C3D and verify all upstream weight tensors.

This verifies the generic feature extractor, not a trained fall SVM or fall score.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sys,json,ast,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'data/source_archives/OriginalBaselines_20261004_r1/runtime'))
import tensorflow as tf
import numpy as np
import h5py
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(1)
notebook=ROOT/'third_party/original_hfd_3dcnn_20261004/HFD_3D_CNN_EA_GitHub_.ipynb'
nb=json.loads(notebook.read_text());source=next(''.join(c['source']) for c in nb['cells'] if 'def create_C3D_model' in ''.join(c['source']))
node=next(x for x in ast.parse(source).body if isinstance(x,ast.FunctionDef) and x.name=='create_C3D_model')
ns={'frame_n':16,'Sequential':tf.keras.Sequential}
ns.update({k:getattr(tf.keras.layers,k) for k in ['Conv3D','MaxPooling3D','ZeroPadding3D','Flatten','Dense','Dropout']})
exec(compile(ast.Module(body=[node],type_ignores=[]),str(notebook),'exec'),ns)
model=ns['create_C3D_model']()
weights=ROOT/'data/source_archives/OriginalCheckpointSearch_20261004_r1/c3d_sports1m_author_linked.h5'
model.load_weights(str(weights))
checks=[]
with h5py.File(weights,'r') as f:
 for tensor in model.weights:
  layer=tensor.name.split('/')[0];key=layer+'/'+tensor.name
  original=f[key][...];loaded=tensor.numpy();assert np.array_equal(original,loaded),tensor.name
  checks.append({'tensor':tensor.name,'shape':list(loaded.shape),'exact_equal':True})
feature_model=tf.keras.Model(model.inputs,model.get_layer('fc6').output)
features=feature_model(np.zeros((1,16,112,112,3),np.float32),training=False).numpy()
assert features.shape==(1,4096) and np.isfinite(features).all()
result=dict(passed=True,author_notebook_sha256=hashlib.sha256(notebook.read_bytes()).hexdigest(),
 weight_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),parameters=model.count_params(),
 tensors_checked=checks,synthetic_feature_shape=list(features.shape),feature_finite=True,
 full_output_classes=model.output_shape[-1],fall_classifier_present=False,fall_evaluation_executed=False,
 training_executed=False,device='CPU',tensorflow=tf.__version__,scope='C3D upstream asset validation only')
dest=ROOT/'docs/internal/2026-10-04_original_checkpoint_search_evidence/c3d_load_verification.json'
dest.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='tensors_checked'}))
