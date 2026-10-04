# Copyright 2018-2019 Open-MMLab. All rights reserved.
# Derived from Privacy X3D-UDA under Apache-2.0; see licenses/Privacy-X3D-UDA-LICENSE.txt.
# Modified for external evaluation; publication paths made portable on 2026-10-04.
work_dir = './work_dirs/x3d_ur_fall_uda/20221225_V08_idm_loss_adaptation_network_three_ratio_solid'
total_epochs = 50
checkpoint_config = {'interval': 2}
log_config = {'interval': 40, 'hooks': [{'type': 'TextLoggerHook'}, {'type': 'TensorboardLoggerHook'}]}
train_dataset_type = 'IDMUdaRawframeDataset'
val_dataset_type = 'UdaRawframeDataset'
data_root = ''
data_root_val = ''
ann_file_train = 'data/kinetics_700/annotation/idm_train_rgb_unsim.txt&&data/kinetics_700/annotation/idm_train_depth_unsim.txt'
ann_file_val = 'data/kinetics_700/annotation/shuffle_test_unsim_depth.txt'
ann_file_test = 'data/kinetics_700/annotation/shuffle_test_unsim_depth.txt'
model = {'type': 'IDMRecognizer3D',
 'backbone': {'type': 'IDMX3D',
              'gamma_w': 1,
              'gamma_b': 2.25,
              'gamma_d': 2.2,
              'pretrained': 'checkpoints/x3d_m_facebook_16x5x1_kinetics400_rgb_20201027-3f42382a.pth',
              'norm_eval': False},
 'cls_head': {'type': 'X3DHead',
              'num_classes': 2,
              'pretrained': 'checkpoints/x3d_m_facebook_16x5x1_kinetics400_rgb_20201027-3f42382a.pth',
              'in_channels': 432,
              'dropout_ratio': 0.4,
              'init_std': 0.01},
 'domain_head': {'type': 'X3DHead',
                 'num_classes': 2,
                 'pretrained': 'checkpoints/x3d_m_facebook_16x5x1_kinetics400_rgb_20201027-3f42382a.pth',
                 'in_channels': 432,
                 'dropout_ratio': 0.4,
                 'init_std': 0.01},
 'train_cfg': {'aux_info': ['domain_label']}}
img_norm_cfg = {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375], 'to_bgr': False}
train_pipeline = [{'type': 'UdaSampleFrames', 'clip_len': 16, 'frame_interval': 5, 'num_clips': 5, 'test_mode': False},
 {'type': 'UdaRawFrameDecode'},
 {'type': 'UdaResize', 'keep_ratio': False, 'scale': (256, 256)},
 {'type': 'UdaNormalize', 'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375], 'to_bgr': False},
 {'type': 'UdaFormatShape', 'input_format': 'NCTHW'},
 {'type': 'Collect', 'keys': ['imgs', 'label', 'domain_label'], 'meta_keys': []},
 {'type': 'ToTensor', 'keys': ['imgs', 'label', 'domain_label']}]
val_pipeline = [{'type': 'SampleFrames', 'clip_len': 16, 'frame_interval': 5, 'num_clips': 5, 'test_mode': True},
 {'type': 'RawFrameDecode'},
 {'type': 'Resize', 'keep_ratio': False, 'scale': (256, 256)},
 {'type': 'Normalize', 'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375], 'to_bgr': False},
 {'type': 'FormatShape', 'input_format': 'NCTHW'},
 {'type': 'Collect', 'keys': ['imgs', 'label', 'domain_label'], 'meta_keys': []},
 {'type': 'ToTensor', 'keys': ['imgs', 'label', 'domain_label']}]
test_pipeline = [{'type': 'SampleFrames', 'clip_len': 16, 'frame_interval': 5, 'num_clips': 5, 'test_mode': True},
 {'type': 'RawFrameDecode'},
 {'type': 'Resize', 'keep_ratio': False, 'scale': (256, 256)},
 {'type': 'Normalize', 'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375], 'to_bgr': False},
 {'type': 'FormatShape', 'input_format': 'NCTHW'},
 {'type': 'Collect', 'keys': ['imgs', 'label', 'domain_label'], 'meta_keys': []},
 {'type': 'ToTensor', 'keys': ['imgs', 'label', 'domain_label']}]
data = {'videos_per_gpu': 1,
 'workers_per_gpu': 0,
 'train_dataloader': {'drop_last': True},
 'val_dataloader': {'videos_per_gpu': 1},
 'test_dataloader': {'videos_per_gpu': 1, 'workers_per_gpu': 0},
 'train': {'type': 'IDMUdaRawframeDataset',
           'ann_file': 'data/kinetics_700/annotation/idm_train_rgb_unsim.txt&&data/kinetics_700/annotation/idm_train_depth_unsim.txt',
           'data_prefix': '',
           'pipeline': [{'type': 'UdaSampleFrames',
                         'clip_len': 16,
                         'frame_interval': 5,
                         'num_clips': 5,
                         'test_mode': False},
                        {'type': 'UdaRawFrameDecode'},
                        {'type': 'UdaResize', 'keep_ratio': False, 'scale': (256, 256)},
                        {'type': 'UdaNormalize',
                         'mean': [123.675, 116.28, 103.53],
                         'std': [58.395, 57.12, 57.375],
                         'to_bgr': False},
                        {'type': 'UdaFormatShape', 'input_format': 'NCTHW'},
                        {'type': 'Collect', 'keys': ['imgs', 'label', 'domain_label'], 'meta_keys': []},
                        {'type': 'ToTensor', 'keys': ['imgs', 'label', 'domain_label']}]},
 'val': {'type': 'UdaRawframeDataset',
         'ann_file': 'data/kinetics_700/annotation/shuffle_test_unsim_depth.txt',
         'data_prefix': '',
         'pipeline': [{'type': 'SampleFrames',
                       'clip_len': 16,
                       'frame_interval': 5,
                       'num_clips': 5,
                       'test_mode': True},
                      {'type': 'RawFrameDecode'},
                      {'type': 'Resize', 'keep_ratio': False, 'scale': (256, 256)},
                      {'type': 'Normalize',
                       'mean': [123.675, 116.28, 103.53],
                       'std': [58.395, 57.12, 57.375],
                       'to_bgr': False},
                      {'type': 'FormatShape', 'input_format': 'NCTHW'},
                      {'type': 'Collect', 'keys': ['imgs', 'label', 'domain_label'], 'meta_keys': []},
                      {'type': 'ToTensor', 'keys': ['imgs', 'label', 'domain_label']}]},
 'test': {'type': 'UdaRawframeDataset',
          'ann_file': '{{ fileDirname }}/../data/fall_processed/RGB/privacy_x3d_external_20261004_r1/cauca100_annotation.txt',
          'data_prefix': '',
          'pipeline': [{'type': 'SampleFrames',
                        'clip_len': 16,
                        'frame_interval': 5,
                        'num_clips': 5,
                        'test_mode': True},
                       {'type': 'RawFrameDecode'},
                       {'type': 'Resize', 'keep_ratio': False, 'scale': (256, 256)},
                       {'type': 'Normalize',
                        'mean': [123.675, 116.28, 103.53],
                        'std': [58.395, 57.12, 57.375],
                        'to_bgr': False},
                       {'type': 'FormatShape', 'input_format': 'NCTHW'},
                       {'type': 'Collect', 'keys': ['imgs', 'label', 'domain_label'], 'meta_keys': []},
                       {'type': 'ToTensor', 'keys': ['imgs', 'label', 'domain_label']}],
          'test_mode': True}}
optimizer = {'type': 'SGD', 'lr': 0.0001, 'momentum': 0.9, 'weight_decay': 0.0001}
optimizer_config = {'grad_clip': {'max_norm': 40, 'norm_type': 2}}
lr_config = {'policy': 'step', 'step': [160, 2000], 'gamma': 0.1}
evaluation = {'interval': 1,
 'gpu_collect': True,
 'metrics': ['top_k_accuracy', 'mean_class_accuracy'],
 'metric_options': {'top_k_accuracy': {'topk': (1, 3)}},
 'save_best': 'top1_acc'}
eval_config = {}
domain_loss_lambda = 0.1
find_unused_parameters = True
test_cfg = {'average_clips': 'score'}
dist_params = {'backend': 'nccl'}
log_level = 'INFO'
load_from = None
resume_from = 'work_dirs/x3d_ur_fall_uda/20221225_V01_idm_loss_adaptation_network_three_auto_baseline/best_top1_acc_epoch_37.pth'
workflow = [('train', 1)]
gpu_ids = [0]
omnisource = False
module_hooks = []
seed = 1024
opencv_num_threads = 0
omp_num_threads = 4
cudnn_benchmark = False
