# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(directory))

import copy
import numpy as np

from fairmotion.utils import utils
from fairmotion.ops import conversions
from fairmotion.ops import quaternion

import motion_utils

from envs import env_humanoid_base

class Env(env_humanoid_base.Env):
    def __init__(self, config):
        super().__init__(config)
        
        self._initialized = False
        self._config = copy.deepcopy(config)
        self._ref_motion = self._base_motion
        self._sensor_lookahead = self._config['state'].get('sensor_lookahead', [0.05, 0.15])
        self._start_time_sampler = self._config.get('start_time_sampler', 'uniform')
        self._start_time = np.zeros(self._num_agent)
        self._repeat_ref_motion = self._config.get('repeat_ref_motion', False)

        assert self._start_time_sampler in ['always_zero', 'uniform']

        if config.get('lazy_creation'):
            if self._verbose:
                print('The environment was created in a lazy fashion.')
                print('The function \"create\" should be called before it')
            return

        self.create()

    def create(self):
        project_dir = self._config['project_dir']
        ref_motion_db = self._config['character'].get('ref_motion_db')
        ref_motion_file = motion_utils.collect_motion_files(project_dir, ref_motion_db)
        
        ''' Load Reference Motion '''

        self._ref_motion_all = []
        self._ref_motion_file_names = []
        for i in range(self._num_agent):
            ref_motion_all, ref_motion_file_names = \
                motion_utils.load_motions(
                    ref_motion_file[i], 
                    None,
                    self._sim_agent[i]._char_info,
                    self._verbose)
            self._ref_motion_all.append(ref_motion_all)
            self._ref_motion_file_names.append(ref_motion_file_names)

        ''' Load Probability of Motion Trajectory '''
        prob_trajectory = self._config['reward'].get('prob_trajectory')
        if prob_trajectory:
            # TODO: Load Model
            self._prob_trajectory = None

        ''' Should call reset after all setups are done '''

        self.reset({'add_noise': False})

        self._initialized = True

        if self._verbose:
            print('----- Humanoid Imitation Environment Created -----')
            for i in range(self._num_agent):
                print('[Agent%d]: state(%d) and action(%d)' \
                      %(i, self.dim_state(i), self.dim_action(i)))
            print('-------------------------------')

    def callback_reset_prev(self, info):
        
        ''' Choose a reference motion randomly whenever reset '''
        
        self._ref_motion, self._ref_motion_idx = \
            self.sample_ref_motion(info.get('ref_motion_id'))
        
        ''' Choose a start time for the current reference motion '''
        
        if 'start_time' in info.keys():
            self._start_time = np.array(info.get('start_time'))
            assert self._start_time.shape[0] == self._num_agent
        else:
            for i in range(self._num_agent):
                if self._start_time_sampler == 'always_zero':
                    self._start_time[i] = 0.0
                elif self._start_time_sampler == 'uniform':
                    self._start_time[i] = \
                        np.random.uniform(0.0, self._ref_motion[i].length())
                else:
                    raise NotImplementedError

    def callback_reset_after(self, info):
        for i in range(self._num_agent):
            self._kin_agent[i].set_pose(
                self._init_poses[i], self._init_vels[i])

    def callback_step_after(self, actions, infos):
        ''' This is necessary to compute the reward correctly '''
        time = self.get_ref_motion_time()
        for i in range(self._num_agent):
            self._kin_agent[i].set_pose(
                self._ref_motion[i].get_pose_by_time(time[i]),
                self._ref_motion[i].get_velocity_by_time(time[i]))

    def print_log_in_step(self):
        if self._verbose and self._end_of_episode:
            print('=================EOE=================')
            print('Reason:', self._end_of_episode_reason)
            print('TIME: (start:%02f) (elapsed:%02f) (time_after_eoe: %02f)'\
                %(self._start_time,
                  self.get_elapsed_time(),
                  self._time_elapsed_after_end_of_episode))
            print('=====================================')
    
    def compute_init_pose_vel(self, info):
        ''' This performs reference-state-initialization (RSI) '''
        init_poses, init_vels = [], []

        for i in range(self._num_agent):
            ''' Set the state of simulated agent by using the state of reference motion '''
            init_pose = self._ref_motion[i].get_pose_by_time(self._start_time[i])
            init_vel = self._ref_motion[i].get_velocity_by_time(self._start_time[i])
            ''' Add noise to the state if necessary '''
            if info.get('add_noise'):
                init_pose, init_vel = \
                    self._base_env.add_noise_to_pose_vel(
                        self._sim_agent[i], init_pose, init_vel)
            init_poses.append(init_pose)
            init_vels.append(init_vel)
        return init_poses, init_vels

    def get_state_by_key(self, idx, key):
        state = []

        ref_motion = self._ref_motion[idx] 
        time = self.get_ref_motion_time()
        
        if key=='body' or key=='body_sim':
            state.append(self.state_body(idx, "sim"))
        elif key=='ref_motion_abs' or key=='ref_motion_rel' or key=='ref_motion_abs_rel' or \
            key=='ref_motion_abs_noff' or key=='ref_motion_rel_noff' or key=='ref_motion_abs_rel_noff':
            ref_motion_abs = True \
                if (key=='ref_motion_abs' or key=='ref_motion_abs_noff' or key=='ref_motion_abs_rel') else False
            ref_motion_rel = True \
                if (key=='ref_motion_rel' or key=='ref_motion_rel_noff' or key=='ref_motion_abs_rel') else False
            facing_frame = True \
                if (key=='ref_motion_abs' or key=='ref_motion_rel' or key=='ref_motion_abs_rel') else False
            poses, vels = [], []
            for dt in self._sensor_lookahead:
                t = np.clip(
                    time[idx] + dt, 
                    0.0, 
                    ref_motion.length())
                poses.append(ref_motion.get_pose_by_time(t))
                vels.append(ref_motion.get_velocity_by_time(t))
            state.append(self.state_imitation(
                idx,
                poses,
                vels,
                include_abs=ref_motion_abs,
                include_rel=ref_motion_rel,
                include_facing_frame=facing_frame))
        elif 'phase_linear' in self._state_choices:
            state.append(
                np.array([
                    self.get_phase(
                        ref_motion, time[idx], mode='linear')]))
        elif 'phase_trigon' in self._state_choices:
            state.append(
                self.get_phase(
                    ref_motion, time[idx], mode='trigon'))
        else:
            raise NotImplementedError

        return np.hstack(state)
    
    def state_body(
        self, 
        idx, 
        agent="sim", 
        type=None, 
        return_stacked=True):
        if agent == "sim":
            agent = self._sim_agent[idx]
        elif agent == "kin":
            agent = self._kin_agent[idx]
        else:
            raise NotImplementedError
        return self._state_body(idx, agent, type, return_stacked)

    def state_task(self, idx):
        sc = self._state_choices.copy()
        sc.remove('body')
        return self.state(idx, sc)

    def state_imitation(
        self, 
        idx,
        poses, 
        vels, 
        include_abs, 
        include_rel,
        include_facing_frame):

        assert len(poses) == len(vels)

        sim_agent = self._sim_agent[idx]
        kin_agent = self._kin_agent[idx]

        R_sim, p_sim = conversions.T2Rp(
            sim_agent.get_facing_transform(self.get_ground_height(idx)))
        R_sim_inv = R_sim.transpose()
        state_sim = self.state_body(idx, agent="sim", return_stacked=False)
        
        state = []
        state_kin_orig = kin_agent.save_states()
        for pose, vel in zip(poses, vels):
            kin_agent.set_pose(pose, vel)
            state_kin = self.state_body(idx, agent="kin", return_stacked=False)
            # Add pos/vel values
            if include_abs:
                state.append(np.hstack(state_kin))
            # Add difference of pos/vel values
            if include_rel:
                for j in range(len(state_sim)):
                    if np.isscalar(state_sim[j]) or len(state_sim[j])==3 or len(state_sim[j])==6: 
                        state.append(state_sim[j]-state_kin[j])
                    elif len(state_sim[j])==4:
                        state.append(
                            self._pb_client.getDifferenceQuaternion(state_sim[j], state_kin[j]))
                    else:
                        raise NotImplementedError
            ''' Add facing frame differences '''
            if include_facing_frame:
                R_kin, p_kin = conversions.T2Rp(
                    kin_agent.get_facing_transform(self.get_ground_height(idx)))
                state.append(np.dot(R_sim_inv, p_kin - p_sim))
                state.append(np.dot(R_sim_inv, kin_agent.get_facing_direction()))
        kin_agent.restore_states(state_kin_orig)

        return np.hstack(state)
    
    def reward_data(self, idx):
        data = {}

        data['sim_root_pQvw'] = self._sim_agent[idx].get_root_state()
        data['sim_link_pQvw'] = self._sim_agent[idx].get_link_states()
        data['sim_joint_pv'] = self._sim_agent[idx].get_joint_states()
        data['sim_facing_frame'] = self._sim_agent[idx].get_facing_transform(self.get_ground_height(idx))
        data['sim_com'], data['sim_com_vel'] = self._sim_agent[idx].get_com_and_com_vel()
        
        data['kin_root_pQvw'] = self._kin_agent[idx].get_root_state()
        data['kin_link_pQvw'] = self._kin_agent[idx].get_link_states()
        data['kin_joint_pv'] = self._kin_agent[idx].get_joint_states()
        data['kin_facing_frame'] = self._kin_agent[idx].get_facing_transform(self.get_ground_height(idx))
        data['kin_com'], data['kin_com_vel'] = self._kin_agent[idx].get_com_and_com_vel()

        return data
    
    def reward_max(self):
        return 1.0
    
    def reward_min(self):
        return 0.0
    
    def get_task_error(self, idx, data_prev, data_next, actions):
        error = {}

        sim_agent = self._sim_agent[idx]
        char_info = sim_agent._char_info

        data = data_next[idx]

        sim_root_p, sim_root_Q, sim_root_v, sim_root_w = data['sim_root_pQvw']
        sim_link_p, sim_link_Q, sim_link_v, sim_link_w = data['sim_link_pQvw']
        sim_joint_p, sim_joint_v = data['sim_joint_pv']
        sim_facing_frame = data['sim_facing_frame']
        R_sim_f, p_sim_f = conversions.T2Rp(sim_facing_frame)
        R_sim_f_inv = R_sim_f.transpose()
        sim_com, sim_com_vel = data['sim_com'], data['sim_com_vel']
        
        kin_root_p, kin_root_Q, kin_root_v, kin_root_w = data['kin_root_pQvw']
        kin_link_p, kin_link_Q, kin_link_v, kin_link_w = data['kin_link_pQvw']
        kin_joint_p, kin_joint_v = data['kin_joint_pv']
        kin_facing_frame = data['kin_facing_frame']
        R_kin_f, p_kin_f = conversions.T2Rp(kin_facing_frame)
        R_kin_f_inv = R_kin_f.transpose()
        kin_com, kin_com_vel = data['kin_com'], data['kin_com_vel']

        indices = range(len(sim_joint_p))

        if self.exist_rew_fn_subterm(idx, 'pose_pos'):
            error['pose_pos'] = 0.0
            for j in indices:
                joint_type = sim_agent.get_joint_type(j)
                if joint_type == self._pb_client.JOINT_FIXED:
                    continue
                elif joint_type == self._pb_client.JOINT_SPHERICAL:
                    dQ = self._pb_client.getDifferenceQuaternion(sim_joint_p[j], kin_joint_p[j])
                    _, diff_pose_pos = self._pb_client.getAxisAngleFromQuaternion(dQ)
                else:
                    diff_pose_pos = sim_joint_p[j] - kin_joint_p[j]
                error['pose_pos'] += char_info.joint_weight[j] * np.dot(diff_pose_pos, diff_pose_pos)
            if len(indices) > 0:
                error['pose_pos'] /= len(indices)

        if self.exist_rew_fn_subterm(idx, 'pose_vel'):
            error['pose_vel'] = 0.0
            for j in indices:
                joint_type = sim_agent.get_joint_type(j)
                if joint_type == self._pb_client.JOINT_FIXED:
                    continue
                else:
                    diff_pose_vel = sim_joint_v[j] - kin_joint_v[j]
                error['pose_vel'] += char_info.joint_weight[j] * np.dot(diff_pose_vel, diff_pose_vel)
            if len(indices) > 0:
                error['pose_vel'] /= len(indices)

        if self.exist_rew_fn_subterm(idx, 'ee'):
            error['ee'] = 0.0
            
            for j in char_info.end_effector_indices:
                sim_ee_local = np.dot(R_sim_f_inv, sim_link_p[j]-p_sim_f)
                kin_ee_local = np.dot(R_kin_f_inv, kin_link_p[j]-p_kin_f)
                diff_pos =  sim_ee_local - kin_ee_local
                error['ee'] += np.dot(diff_pos, diff_pos)

            if len(char_info.end_effector_indices) > 0:
                error['ee'] /= len(char_info.end_effector_indices)

        if self.exist_rew_fn_subterm(idx, 'root'):
            diff_root_p = sim_root_p - kin_root_p
            _, diff_root_Q = self._pb_client.getAxisAngleFromQuaternion(
                self._pb_client.getDifferenceQuaternion(sim_root_Q, kin_root_Q))
            diff_root_v = sim_root_v - kin_root_v
            diff_root_w = sim_root_w - kin_root_w
            error['root'] = 1.0 * np.dot(diff_root_p, diff_root_p) + \
                            0.1 * np.dot(diff_root_Q, diff_root_Q) + \
                            0.01 * np.dot(diff_root_v, diff_root_v) + \
                            0.001 * np.dot(diff_root_w, diff_root_w)

        if self.exist_rew_fn_subterm(idx, 'com'):
            diff_com = np.dot(R_sim_f_inv, sim_com-p_sim_f) - np.dot(R_kin_f_inv, kin_com-p_kin_f)
            diff_com_vel = np.dot(R_sim_f_inv, sim_com_vel) - np.dot(R_kin_f_inv, kin_com_vel)
            error['com'] = 1.0 * np.dot(diff_com, diff_com) + \
                           0.1 * np.dot(diff_com_vel, diff_com_vel)

        if self.exist_rew_fn_subterm(idx, 'prob_trajectory'):
            # TODO
            X = None
            error['prob_trajectory'] = self._prob_trajectory(X)

        return error

    def inspect_end_of_episode(self):
        eoe_reason = super().inspect_end_of_episode()
        
        cur_time = self.get_current_time()
        for i in range(self._num_agent):
            name = self._sim_agent[i].get_name()
            if "ref_motion_end" in self._early_term_choices:
                check = cur_time[i] >= self._ref_motion[i].length()
                if check: eoe_reason.append('[%s] end_of_motion'%self._sim_agent[i].get_name())    
            if "root_mismatch_orientation" in self._early_term_choices or \
               "root_mismatch_position" in self._early_term_choices:
                p1, Q1, v1, w1 = self._sim_agent[i].get_root_state()
                p2, Q2, v2, w2 = self._kin_agent[i].get_root_state()
                ''' TODO: remove pybullet Q utils '''
                if "root_mismatch_orientation" in self._early_term_choices:
                    dQ = self._pb_client.getDifferenceQuaternion(Q1, Q2)
                    _, diff = self._pb_client.getAxisAngleFromQuaternion(dQ)
                    # Defalult threshold is 60 degrees
                    thres = self._config['early_term'].get('root_mismatch_orientation_thres', 1.0472)
                    check = diff > thres
                    if check: eoe_reason.append('[%s]:root_mismatch_orientation'%name)
                if "root_mismatch_position" in self._early_term_choices:
                    diff = np.linalg.norm(p2-p1)
                    thres = self._config['early_term'].get('root_mismatch_position_thres', 0.5)
                    check = diff > thres
                    if check: eoe_reason.append('[%s]:root_mismatch_position'%name)
        return eoe_reason        

    def get_ref_motion_time(self):
        cur_time = self.get_current_time()
        if self._repeat_ref_motion:
            for i in range(self._num_agent):
                motion_length = self._ref_motion[i].length()
                while cur_time[i] > motion_length:
                    cur_time[i] -= motion_length
        return cur_time

    def get_current_time(self):
        return self._start_time + self.get_elapsed_time()

    def sample_ref_motion(self, indices=None):
        ref_indices = []
        ref_motions = []
        for i in range(self._num_agent):
            if indices is None:
                idx = np.random.randint(len(self._ref_motion_all[i]))
            else:
                idx = indices[i]
            ref_indices.append(idx)
            ref_motions.append(self._ref_motion_all[i][idx])
        if self._verbose:
            print('Ref. motions selected: ', ref_indices)
        return ref_motions, ref_indices

    def get_phase(self, motion, elapsed_time, mode='linear', **kwargs):
        if mode == 'linear':
            return elapsed_time / motion.length()
        elif mode == 'trigon':
            period = kwargs.get('period', 1.0)
            theta = 2*np.pi * elapsed_time / period
            return np.array([np.cos(theta), np.sin(theta)])
        else:
            raise NotImplementedError

    def check_end_of_motion(self, idx):
        cur_time = self.get_current_time()
        return cur_time[idx] >= self._ref_motion[idx].length()

    def render(self, rm):
        super().render(rm)
