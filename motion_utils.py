# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle
import gzip
import re
import random

from fairmotion.core.motion import Motion
from fairmotion.core.velocity import MotionWithVelocity
from fairmotion.data import bvh
from fairmotion.utils import utils


def load_motions(motion_files, skel, char_info, verbose):
    assert motion_files is not None
    motion_all = []
    motion_file_names = []
    for names in motion_files:
        head, tail = os.path.split(names)
        motion_file_names.append(tail)
    if len(motion_files) > 0:
        if isinstance(motion_files[0], str):
            motion_dict = {}
            for i, file in enumerate(motion_files):
                ''' If the same file is already loaded, do not load again for efficiency'''
                if file in motion_dict:
                    m = motion_dict[file]
                else:
                    if file.endswith('bvh'):
                        if skel is not None:
                            m = bvh.load(motion=Motion(name=file, skel=skel),
                                         file=file,
                                         scale=1.0, 
                                         load_skel=False,
                                         v_up_skel=char_info.v_up, 
                                         v_face_skel=char_info.v_face, 
                                         v_up_env=char_info.v_up_env)
                        else:
                            m = bvh.load(file=file,
                                         scale=1.0, 
                                         v_up_skel=char_info.v_up, 
                                         v_face_skel=char_info.v_face, 
                                         v_up_env=char_info.v_up_env)
                        m = MotionWithVelocity.from_motion(m)
                    elif file.endswith('bin'):
                        m = pickle.load(open(file, "rb"))
                    elif file.endswith('gzip') or file.endswith('gz'):
                        with gzip.open(file, "rb") as f:
                            m = pickle.load(f)
                    else:
                        raise Exception('Unknown Motion File Type')
                    if verbose: 
                        print('Loaded: %s'%file)
                motion_all.append(m)
        elif isinstance(motion_files[0], MotionWithVelocity):
            motion_all = motion_files
        else:
            raise Exception('Unknown Type for Reference Motion')

    return motion_all, motion_file_names

def collect_motion_files(project_dir, ref_motion_db):
    ref_motion_file = []
    for i, mdb in enumerate(ref_motion_db):
        motions = []
        if mdb.get('cluster_info'):
            ''' Read reference motions based on the cluster labels '''
            assert mdb.get('data') is None, \
                'This should not be specified when cluster_info is used'
            dir = mdb['cluster_info'].get('dir')
            label_file = mdb['cluster_info'].get('label_file')
            sample_id = mdb['cluster_info'].get('sample_id')
            labels = {}
            assert label_file
            if project_dir:
                label_file = os.path.join(project_dir, label_file)
            with open(label_file, 'r') as file:
                for line in file:
                    l = re.split('[\t|\n|,|:| ]+', line)
                    id, rank, score, filename = int(l[0]), int(l[1]), float(l[2]), str(l[3])
                    if id not in labels.keys(): labels[id] = []
                    labels[id].append({'rank': rank, 'socre': score, 'filename': filename})
            num_cluster = len(labels.keys())
            for j in range(num_cluster):
                if sample_id and j!=sample_id:
                    continue
                for label in labels[j]:
                    if project_dir:
                        file = os.path.join(project_dir, dir, label['filename'])
                    motions.append(file)
        else:
            ''' Read reference motions from the specified list of files and dirs '''
            ref_motion_data = mdb.get('data')
            motions = []
            if ref_motion_data.get('file'):
                motions += ref_motion_data.get('file')
            if ref_motion_data.get('dir'):
                for d in ref_motion_data.get('dir'):
                    if project_dir:
                        d = os.path.join(project_dir, d)
                    motions += utils.files_in_dir(d, ext=".bvh", sort=True)
            if project_dir:
                for j in range(len(motions)):
                    motions[j] = os.path.join(project_dir, motions[j])
        ''' 
        If num_sample is specified, we use only num_sample motions 
        from the entire reference motions. 
        'random' chooses randomly, 'top' chooses the first num_sample
        '''
        num_sample = mdb.get('num_sample')
        if num_sample:
            sample_method = mdb.get('sample_method')
            if sample_method == 'random':
                motions = random.choices(motions, k=num_sample)
            elif sample_method == 'top':
                motions = motions[:num_sample]
            else:
                raise NotImplementedError
        ref_motion_file.append(motions)
    return ref_motion_file


# for i, ref_motion in enumerate(self._ref_motion_all):
#     for j, m in enumerate(ref_motion):
#         _, p = conversions.T2Rp(m.get_pose_by_frame(0).get_root_transform())
#         v = np.zeros(3)
#         if i==0:
#             v[1] = -p[1] - 2.0
#             R = conversions.Az2R(-0.5*np.pi)
#         else:
#             v[1] = -p[1] + 2.0
#             R = conversions.Az2R(0.5*np.pi)
#         T = conversions.Rp2T(R, v)
#         motion.transform(m, T, 0)
#         bvh.save(m, "data/temp/"+self._ref_motion_file_names[i][j]+"_edited.bvh", verbose=True)
# exit(0)