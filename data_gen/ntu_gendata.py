import argparse
import json
import pickle
from tqdm import tqdm
import sys
from numpy.lib.format import open_memmap

sys.path.extend(['../'])
if __package__:
    from .preprocess import pre_normalization_ntu_umurl
else:
    from preprocess import pre_normalization_ntu_umurl

UMURL_PREPROCESS_COMMIT = '744278b8ad6ba3fc2290cd2469c4bf315c571935'
UMURL_PREPROCESS_SOURCE = (
    'https://github.com/HuiGuanLab/UmURL/blob/'
    + UMURL_PREPROCESS_COMMIT
    + '/data_gen/preprocess.py'
)

NTU60_TRAINING_SUBJECTS = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
]

NTU120_TRAINING_SUBJECTS = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38, 
    45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78,80, 81, 82, 
    83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103
]
training_setups = [ 2, 4,  6, 8, 10, 12, 14,  16,  18,  20, 22, 24, 26, 28, 30, 32]

training_cameras = [2, 3]
max_body_true = 2
max_body_kinect = 4
num_joint = 25
max_frame = 300

EXPECTED_COUNTS = {
    ('ntu60', 'xsub', 'train'): 40091,
    ('ntu60', 'xsub', 'val'): 16487,
    ('ntu60', 'xview', 'train'): 37646,
    ('ntu60', 'xview', 'val'): 18932,
    ('ntu120', 'xsub', 'train'): 63026,
    ('ntu120', 'xsub', 'val'): 50919,
    ('ntu120', 'xsetup', 'train'): 54471,
    ('ntu120', 'xsetup', 'val'): 59474,
}

import numpy as np
import os


def read_skeleton_filter(file):
    with open(file, 'r') as f:
        skeleton_sequence = {}
        skeleton_sequence['numFrame'] = int(f.readline())
        skeleton_sequence['frameInfo'] = []
        # num_body = 0
        for t in range(skeleton_sequence['numFrame']):
            frame_info = {}
            frame_info['numBody'] = int(f.readline())
            frame_info['bodyInfo'] = []

            for m in range(frame_info['numBody']):
                body_info = {}
                body_info_key = [
                    'bodyID', 'clipedEdges', 'handLeftConfidence',
                    'handLeftState', 'handRightConfidence', 'handRightState',
                    'isResticted', 'leanX', 'leanY', 'trackingState'
                ]
                body_info = {
                    k: float(v)
                    for k, v in zip(body_info_key, f.readline().split())
                }
                body_info['numJoint'] = int(f.readline())
                body_info['jointInfo'] = []
                for v in range(body_info['numJoint']):
                    joint_info_key = [
                        'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                        'orientationW', 'orientationX', 'orientationY',
                        'orientationZ', 'trackingState'
                    ]
                    joint_info = {
                        k: float(v)
                        for k, v in zip(joint_info_key, f.readline().split())
                    }
                    body_info['jointInfo'].append(joint_info)
                frame_info['bodyInfo'].append(body_info)
            skeleton_sequence['frameInfo'].append(frame_info)

    return skeleton_sequence


def get_nonzero_std(s):  # tvc
    index = s.sum(-1).sum(-1) != 0  # select valid frames
    s = s[index]
    if len(s) != 0:
        s = s[:, :, 0].std() + s[:, :, 1].std() + s[:, :, 2].std()  # three channels
    else:
        s = 0
    return s


def read_xyz(file, max_body=4, num_joint=25):  # 取了前两个body
    seq_info = read_skeleton_filter(file)
    data = np.zeros((max_body, seq_info['numFrame'], num_joint, 3))
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[m, n, j, :] = [v['x'], v['y'], v['z']]
                else:
                    pass

    # select two max energy body
    energy = np.array([get_nonzero_std(x) for x in data])
    index = energy.argsort()[::-1][0:max_body_true]
    data = data[index]

    data = data.transpose(3, 1, 2, 0)
    return data


def gendata(data_path, out_path, ignored_sample_path=None, benchmark='xview', part='eval',
            dataset='ntu60', normalization_profile='official_umurl', expected_count=None,
            modalities=('joint',)):
    Bone = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                     (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                     (18, 17), (19, 18), (20, 19), (21, 21), (22, 23), (23, 8), (24, 25), (25, 12)]
    
    if ignored_sample_path != None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [
                line.strip() + '.skeleton' for line in f.readlines()
            ]
    else:
        ignored_samples = []
    sample_name = []
    sample_label = []
    for filename in sorted(os.listdir(data_path)):
        if filename in ignored_samples:
            continue
        action_class = int(
            filename[filename.find('A') + 1:filename.find('A') + 4])
        subject_id = int(
            filename[filename.find('P') + 1:filename.find('P') + 4])
        camera_id = int(
            filename[filename.find('C') + 1:filename.find('C') + 4])
        setup_id = int(
            filename[filename.find('S') + 1:filename.find('S') + 4])

        if benchmark == 'xview':
            istraining = (camera_id in training_cameras)
        elif benchmark == 'xsub':
            training_subjects = (NTU60_TRAINING_SUBJECTS if dataset == 'ntu60'
                                 else NTU120_TRAINING_SUBJECTS)
            istraining = (subject_id in training_subjects)
        elif benchmark == 'xsetup':
            istraining = (setup_id in training_setups)
        else:
            raise ValueError()

        if part == 'train':
            issample = istraining
        elif part == 'val':
            issample = not (istraining)
        else:
            raise ValueError()

        if issample:
            sample_name.append(filename)
            sample_label.append(action_class - 1)

    if expected_count is not None and len(sample_name) != expected_count:
        raise RuntimeError('official split count mismatch: got {}, expected {}'.format(
            len(sample_name), expected_count))

    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)

    fl = open_memmap(
        '{}/{}_num_frame.npy'.format(out_path, part),
        dtype='int',
        mode='w+',
        shape=(len(sample_label),))

    shape = (len(sample_label), 3, max_frame, num_joint, max_body_true)
    arrays = {}
    for modality in modalities:
        arrays[modality] = open_memmap(
            os.path.join(out_path, '{}_data_{}.npy'.format(part, modality)),
            dtype='float32',
            mode='w+',
            shape=shape)

    for i, s in enumerate(tqdm(sample_name)):
        data = read_xyz(os.path.join(data_path, s), max_body=max_body_kinect, num_joint=num_joint)
        frame_count = min(data.shape[1], max_frame)
        data = data[:, :frame_count, :, :]
        if 'joint' in arrays:
            arrays['joint'][i, :, :frame_count, :, :] = data
        if 'motion' in arrays:
            motion = np.zeros_like(data)
            motion[:, :-1, :, :] = data[:, 1:, :, :] - data[:, :-1, :, :]
            arrays['motion'][i, :, :frame_count, :, :] = motion
        if 'bone' in arrays:
            bone = np.zeros_like(data)
            for v1, v2 in Bone:
                bone[:, :, v1 - 1, :] = data[:, :, v1 - 1, :] - data[:, :, v2 - 1, :]
            arrays['bone'][i, :, :frame_count, :, :] = bone
        fl[i] = data.shape[1] # num_frame

    if normalization_profile == 'official_umurl':
        if 'joint' in arrays:
            normalized = pre_normalization_ntu_umurl(arrays['joint'])
            if not np.shares_memory(normalized, arrays['joint']):
                arrays['joint'][:] = normalized
    elif normalization_profile != 'ntu_standard':
        raise ValueError('normalization_profile must be official_umurl or ntu_standard')
    for array in arrays.values():
        array.flush()
    fl.flush()
    return len(sample_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NTU-RGB-D Data Converter.')
    parser.add_argument('--dataset', choices=('ntu60', 'ntu120'), required=True)
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--ignored-sample-path')
    parser.add_argument('--out-folder', required=True)
    parser.add_argument('--protocol', choices=('xsub', 'xview', 'xsetup'), action='append')
    parser.add_argument('--split', choices=('train', 'val'), action='append')
    parser.add_argument('--normalization-profile',
                        choices=('official_umurl', 'ntu_standard'), default='official_umurl')
    parser.add_argument('--modality', choices=('joint', 'motion', 'bone'), action='append',
                        help='output modality; repeat to generate multiple (default: joint)')
    parser.add_argument('--overwrite', action='store_true')
    arg = parser.parse_args()
    protocols = arg.protocol or (['xsub', 'xview'] if arg.dataset == 'ntu60' else ['xsub', 'xsetup'])
    parts = arg.split or ['train', 'val']
    modalities = tuple(dict.fromkeys(arg.modality or ['joint']))
    allowed_protocols = {'ntu60': {'xsub', 'xview'}, 'ntu120': {'xsub', 'xsetup'}}
    invalid_protocols = set(protocols) - allowed_protocols[arg.dataset]
    if invalid_protocols:
        raise ValueError('invalid protocol(s) for {}: {}'.format(
            arg.dataset, ', '.join(sorted(invalid_protocols))))
    for b in protocols:
        for p in parts:
            out_path = os.path.join(arg.out_folder, b)
            os.makedirs(out_path, exist_ok=True)
            targets = [os.path.join(out_path, '{}_data_{}.npy'.format(p, modality))
                       for modality in modalities]
            targets.extend(os.path.join(out_path, '{}_{}'.format(p, suffix))
                           for suffix in ('num_frame.npy', 'label.pkl'))
            existing = [path for path in targets if os.path.exists(path)]
            if existing and not arg.overwrite:
                raise FileExistsError('refusing to overwrite: ' + ', '.join(existing))
            print(b, p)
            expected = EXPECTED_COUNTS.get((arg.dataset, b, p))
            gendata(
                arg.data_path,
                out_path,
                arg.ignored_sample_path,
                benchmark=b,
                part=p,
                dataset=arg.dataset,
                normalization_profile=arg.normalization_profile,
                expected_count=expected,
                modalities=modalities)
    manifest = {
        'dataset': arg.dataset,
        'data_path': os.path.abspath(arg.data_path),
        'protocols': protocols,
        'splits': parts,
        'normalization_profile': arg.normalization_profile,
        'modalities': modalities,
    }
    if arg.normalization_profile == 'official_umurl':
        manifest['normalization_contract'] = {
            'source': UMURL_PREPROCESS_SOURCE,
            'commit': UMURL_PREPROCESS_COMMIT,
            'center_joint': 1,
            'zaxis_alignment': None,
            'xaxis_vector': [8, 4],
            'xaxis_target': [1, 0, 0],
        }
    with open(os.path.join(arg.out_folder, 'preprocess_manifest.json'), 'w') as stream:
        json.dump(manifest, stream, indent=2)
        stream.write('\n')
