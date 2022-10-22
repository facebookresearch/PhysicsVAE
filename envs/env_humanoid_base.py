# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(directory))


import numpy as np
import copy
from enum import Enum
from collections import deque

from fairmotion.ops import conversions
from fairmotion.ops import math
from fairmotion.ops import quaternion
from fairmotion.core.motion import Pose
from fairmotion.core.velocity import MotionWithVelocity
from fairmotion.data import bvh

from envs import env_humanoid_tracking
import sim_agent

from abc import ABCMeta, abstractmethod

profile = False
if profile:
    from fairmotion.viz.utils import TimeChecker
    tc = TimeChecker()

class Env(metaclass=ABCMeta):
    class ActionMode(Enum):
        Absolute=0 # Use an absolute posture as an action
        Relative=1 # Use a relative posture from a reference posture as an action
        @classmethod
        def from_string(cls, string):
            if string=="absolute": return cls.Absolute
            if string=="relative": return cls.Relative
            raise NotImplementedError
    def __init__(self, config):
        config           = copy.deepcopy(config)
        project_dir      = config['project_dir']
        sim_char_file    = config['character'].get('sim_char_file')
        base_motion_file = config['character'].get('base_motion_file')
        ref_motion_file  = config['character'].get('ref_motion_file')
        joint_correction = config['character'].get('joint_correction')
        environment_file = config['character'].get('environment_file')

        ''' Create a base tracking environment '''

        self._base_env = env_humanoid_tracking.Env(config)
        
        ''' Append project_dir to the given file path '''
        
        for i in range(len(sim_char_file)):
            base_motion_file[i] = os.path.join(project_dir, base_motion_file[i])
        if environment_file is not None:
            for i in range(len(environment_file)):
                environment_file[i] = os.path.join(project_dir, environment_file[i])

        self._ground = self._base_env._ground
        self._pb_client = self._base_env._pb_client
        self._fps_con = self._base_env._fps_con
        self._dt_con = self._base_env._dt_con

        ''' Copy some of frequently used attributes from the base environemnt '''
        self._num_agent = self._base_env._num_agent
        assert self._num_agent == len(base_motion_file)
        self._sim_agent = self._base_env._agent
        self._v_up = self._base_env._v_up
        self._v_up_str = self._base_env._v_up_str

        ''' State '''
        self._state_choices = config['state']['choices']
        self._state_body_type = config['state']['body_type']

        ''' Early Terminations '''
        self._early_term_choices = config['early_term']['choices']
        assert len(self._early_term_choices) > 0, "No early termination option is specified"

        '''
        Reward computation is designed in a structured manner so that it allows 
        fast prototyping and diverse experiments by avoding hardcoded rewards.
        For example, let's say we have 5 reward funtions and 2 simulated agents.
        The example below shows that the first agent uses 2nd reward funtions,
        and the second agent uses 5th reward funtions.
        reward_fn_def: {'rew_fn_name_1':rew_fn_def_1, 'rew_fn_name_2':rew_fn_def_2, ...}
        reward_fn_map: ['rew_fn_name_2', 'rew_fn_name_5']
        '''

        ''' 
        The definitions of all avaialble reward functions, 
        whose keys are the names of the funtions
        '''
        self._reward_fn_def = config['reward']['fn_def']
        ''' 
        The mapping telling which reward is currently asigned to an agent.
        By changing this mapping, the agent can use different rewards on the fly.
        Please note that corresponding task errors should be correctly computed 
        before computing the actual reward functions, refer to 'get_task_error'.
        '''
        self._reward_fn_map = config['reward']['fn_map']
        ''' 
        For each reward function, we collect all names of individual reward term.
        This is helpuful to save time by computing relevant error terms 
        that are necessary in the current reward function. 
        '''
        self._reward_fn_subterm_names = {}
        for fn_name, fn_def in self._reward_fn_def.items():
            self._reward_fn_subterm_names[fn_name] = self.get_reward_fn_subterms_in_use(fn_def)
        self._reward_fn_subterms_in_use = [
            self._reward_fn_subterm_names[self._reward_fn_map[i]] \
            for i in range(self._num_agent)]
        
        # self._reward_fn_subterms_in_use = [self.get_reward_fn_subterms_in_use(
        #     self._reward_fn_def[self._reward_fn_map[i]]) for i in range(self._num_agent)]
        self._rew_data_prev = None

        '''
        Check the existence of reward definitions, which are defined in our reward map
        '''
        assert len(self._reward_fn_map) == self._num_agent
        for key in self._reward_fn_map:
            assert key in self._reward_fn_def.keys()

        self._verbose = config['verbose']

        if "falldown_by_height" in self._early_term_choices:
            self._et_height_min = config['early_term']['height_min']

        if "low_reward" in self._early_term_choices:
            self._et_low_reward_thres = config['early_term'].get('low_reward_thres', 0.1)
            self._rew_queue = self._num_agent * [None]
            for i in range(self._num_agent):
                self._rew_queue[i] = deque(
                    maxlen=int(self._fps_con * config['early_term'].get('low_reward_duration', 1.0)))

        if "high_reward_sum" in self._early_term_choices:
            self._et_high_rew_sum_thres = config['early_term'].get('high_rew_sum_thres', np.inf)
            self._rew_sum = np.zeros(self._num_agent)
        
        ''' The environment automatically terminates after 'sim_window' seconds '''
        if "sim_window" in self._early_term_choices:
            self._sim_window_time = config['early_term']['sim_window_time']
        ''' 
        The environment continues for "eoe_margin" seconds after end-of-episode is set by TRUE.
        This is useful for making the controller work for boundaries of reference motions
        '''
        self._eoe_margin = config['early_term']['eoe_margin']

        action_type = config['action'].get('type')
        if action_type:
            self._action_type = Env.ActionMode.from_string(action_type)
        else:
            self._action_type = None

        ''' Setup Joint Correction info '''

        self._joint_correction_info = self._num_agent*[None]
        if joint_correction:
            assert len(joint_correction) == self._num_agent
            for i in range(self._num_agent):
                name = joint_correction[i]['info']['name']
                axis = joint_correction[i]['info']['axis']
                assert len(name)==len(axis)
                info = []
                for j in range(len(name)):
                    info.append((
                        self._sim_agent[i]._char_info.joint_idx[name[j]], 
                        np.array(axis[j])))
                self._joint_correction_info[i] = info

        ''' Base motion defines the initial posture (like t-pose) '''

        self._base_motion = []
        for i in range(self._num_agent):
            m = bvh.load(file=base_motion_file[i],
                         motion=MotionWithVelocity(),
                         scale=1.0, 
                         load_skel=True,
                         load_motion=True,
                         v_up_skel=self._sim_agent[i]._char_info.v_up, 
                         v_face_skel=self._sim_agent[i]._char_info.v_face, 
                         v_up_env=self._sim_agent[i]._char_info.v_up_env)
            m = MotionWithVelocity.from_motion(m)
            self._base_motion.append(m)

        ''' Create Kinematic Agents '''
        
        self._kin_agent = []
        for i in range(self._num_agent):
            self._kin_agent.append(
                sim_agent.SimAgent(pybullet_client=self._base_env._pb_client, 
                                   model_file=sim_char_file[i],
                                   char_info=self._sim_agent[i]._char_info,
                                   ref_scale=self._sim_agent[i]._ref_scale,
                                   ref_height_fix=self._sim_agent[i]._ref_height_fix,
                                   self_collision=self._sim_agent[i]._self_collision,
                                   kinematic_only=True,
                                   verbose=self._base_env._verbose))

        ''' 
        Define the action space of this environment.
        Here I used a 'normalizer' where 'real' values correspond to joint angles,
        and 'norm' values correspond to the output value of NN policy.
        The reason why it is used is that NN policy somtimes could output values that
        are within much larger or narrow range than we need for the environment.
        For example, if we apply tanh activation function at the last layer of NN,
        the output are always within (-1, 1), but we need bigger values for joint angles 
        because 1 corresponds only to 57.3 degree.
        '''
        self._action_space = []
        self._use_base_residual_linear_force = \
            config['action'].get('use_base_residual_linear_force', False)
        self._base_residual_linear_force_frame = \
            config['action'].get('base_residual_linear_force_frame', "base")
        self._use_base_residual_angular_force = \
            config['action'].get('use_base_residual_angular_force', False)
        self._base_residual_angular_force_frame = \
            config['action'].get('base_residual_angular_force_frame', "base")
        for i in range(self._num_agent):
            act_space = {}
            ''' Action space definition for the target pose '''
            dim_pose = self._sim_agent[i].get_num_dofs()
            if self._sim_agent[i]._actuation==sim_agent.SimAgent.Actuation.NONE:
                pass
            elif self._sim_agent[i]._actuation==sim_agent.SimAgent.Actuation.TQ:
                real_val_max = np.array(self._sim_agent[i]._act_max_forces).flatten()
                real_val_min = -1.0 * real_val_max
                norm_val_max = config['action']['range_max_pol']*np.ones(dim_pose)
                norm_val_min = config['action']['range_min_pol']*np.ones(dim_pose)
                act_space['torque'] = math.Normalizer(
                    real_val_max=real_val_max,
                    real_val_min=real_val_min,
                    norm_val_max=norm_val_max,
                    norm_val_min=norm_val_min,
                    apply_clamp=config['action'].get('apply_clamp', True))
            else:
                dim_pose = self._sim_agent[i].get_num_dofs()
                real_val_max = config['action']['range_max']*np.ones(dim_pose)
                real_val_min = config['action']['range_min']*np.ones(dim_pose)
                norm_val_max = config['action']['range_max_pol']*np.ones(dim_pose)
                norm_val_min = config['action']['range_min_pol']*np.ones(dim_pose)
                act_space['target_pose'] = math.Normalizer(
                    real_val_max=real_val_max,
                    real_val_min=real_val_min,
                    norm_val_max=norm_val_max,
                    norm_val_min=norm_val_min,
                    apply_clamp=config['action'].get('apply_clamp', True))
            ''' Action space definition for the residual forces '''
            if self._use_base_residual_angular_force:
                real_val_max = np.array(config['action']['range_max_angular_force'])
                real_val_min = np.array(config['action']['range_min_angular_force'])
                norm_val_max = np.array(config['action']['range_max_pol_angular_force'])
                norm_val_min = np.array(config['action']['range_min_pol_angular_force'])
                act_space['base_residual_angular_force'] = math.Normalizer(
                    real_val_max=real_val_max,
                    real_val_min=real_val_min,
                    norm_val_max=norm_val_max,
                    norm_val_min=norm_val_min,
                    apply_clamp=config['action'].get('apply_clamp', True))
            if self._use_base_residual_linear_force:
                real_val_max = np.array(config['action']['range_max_linear_force'])
                real_val_min = np.array(config['action']['range_min_linear_force'])
                norm_val_max = np.array(config['action']['range_max_pol_linear_force'])
                norm_val_min = np.array(config['action']['range_min_pol_linear_force'])
                act_space['base_residual_linear_force'] = math.Normalizer(
                    real_val_max=real_val_max,
                    real_val_min=real_val_min,
                    norm_val_max=norm_val_max,
                    norm_val_min=norm_val_min,
                    apply_clamp=config['action'].get('apply_clamp', True))

            self._action_space.append(act_space)

        ''' 
        Any necessary information needed for training this environment.
        This can be set by calling "set_learning_info". 
        '''
        self._learning_info = {}

        self.add_noise = config['add_noise']

        self._config = config

    def get_ground(self):
        return self._base_env._ground

    def update_reward_fn_map(self, idx, rew_fn_name):
        assert rew_fn_name in self._reward_fn_def.keys()
        self._reward_fn_map[idx] = rew_fn_name
        self._reward_fn_subterms_in_use[idx] = \
            self._reward_fn_subterm_names[self._reward_fn_map[idx]]

    def exist_rew_fn_subterm(self, idx, name):
        return name in self._reward_fn_subterms_in_use[idx]

    def action_range(self, idx):
        norm_val_min = np.array([])
        norm_val_max = np.array([])
        if self._action_space[idx].get('target_pose'):
            norm_val_min_new = self._action_space[idx]['target_pose'].norm_val_min
            norm_val_max_new = self._action_space[idx]['target_pose'].norm_val_max
            norm_val_min = np.hstack([norm_val_min_new, norm_val_min])
            norm_val_max = np.hstack([norm_val_max_new, norm_val_max])
        if self._action_space[idx].get('torque'):
            norm_val_min_new = self._action_space[idx]['torque'].norm_val_min
            norm_val_max_new = self._action_space[idx]['torque'].norm_val_max
            norm_val_min = np.hstack([norm_val_min_new, norm_val_min])
            norm_val_max = np.hstack([norm_val_max_new, norm_val_max])
        if self._action_space[idx].get('base_residual_angular_force'):
            norm_val_min_new = self._action_space[idx]['base_residual_angular_force'].norm_val_min
            norm_val_max_new = self._action_space[idx]['base_residual_angular_force'].norm_val_max
            norm_val_min = np.hstack([norm_val_min_new, norm_val_min])
            norm_val_max = np.hstack([norm_val_max_new, norm_val_max])
        if self._action_space[idx].get('base_residual_linear_force'):
            norm_val_min_new = self._action_space[idx]['base_residual_linear_force'].norm_val_min
            norm_val_max_new = self._action_space[idx]['base_residual_linear_force'].norm_val_max
            norm_val_min = np.hstack([norm_val_min_new, norm_val_min])
            norm_val_max = np.hstack([norm_val_max_new, norm_val_max])
        return norm_val_min, norm_val_max

    def dim_action(self, idx):
        dim = 0
        if self._action_space[idx].get('target_pose'):
            dim += self._action_space[idx]['target_pose'].dim
        if self._action_space[idx].get('torque'):
            dim += self._action_space[idx]['torque'].dim
        if self._action_space[idx].get('base_residual_angular_force'):
            dim += self._action_space[idx]['base_residual_angular_force'].dim
        if self._action_space[idx].get('base_residual_linear_force'):
            dim += self._action_space[idx]['base_residual_linear_force'].dim
        return dim

    def dim_state(self, idx):
        return len(self.state(idx))

    def dim_state_body(self, idx):
        return len(self.state_body(idx))

    def dim_state_task(self, idx):
        return len(self.state_task(idx))

    def set_learning_info(self, info):
        self._learning_info = info

    def update_learning_info(self, info):
        self._learning_info.update(info)

    def agent_avg_position(self, agents=None):
        if agents is None: agents=self._sim_agent
        return np.mean([(agent.get_root_state())[0] for agent in agents], axis=0)

    def agent_ave_facing_position(self, agents=None):
        if agents is None: agents=self._sim_agent
        return np.mean([agent.get_facing_position(self.get_ground_height()) for agent in agents], axis=0)

    def throw_obstacle(self):
        size = np.random.uniform(0.1, 0.3, 3)
        p = self.agent_avg_position()
        self._base_env.throw_obstacle(size, p)

    def split_action(self, action):
        assert len(action)%self._num_agent == 0
        dim_action = len(action)//self._num_agent
        actions = []
        idx = 0
        for i in range(self._num_agent):
            actions.append(action[idx:idx+dim_action])
            idx += dim_action
        return actions

    def compute_target_pose(self, idx, action):
        '''
        We assume that the given action is already denormalized
        '''

        agent = self._sim_agent[idx]
        char_info = agent._char_info
        
        ''' the current posture should be deepcopied because action will modify it '''
        if self._action_type == Env.ActionMode.Relative:
            ref_pose = copy.deepcopy(self.get_current_pose_from_motion(idx))
        else:
            ref_pose = copy.deepcopy(self._base_motion[idx].get_pose_by_frame(0))
        
        '''
        Collect action values (axis-angle) and its base xforms
        '''
        dof_cnt = 0
        As = []
        Ts_base = []
        bvh_map_indices = []
        for j in agent._joint_indices:
            joint_type = agent.get_joint_type(j)
            ''' Fixed joint will not be affected '''
            if joint_type == self._pb_client.JOINT_FIXED:
                continue
            ''' If the joint do not have correspondance, use the reference posture itself'''
            if char_info.bvh_map[j] == None:
                continue
            # if self._action_type == Env.ActionMode.Relative:
            #     T = ref_pose.get_transform(char_info.bvh_map[j], local=True)
            # elif self._action_type == Env.ActionMode.Absolute:
            #     T = ref_pose.skel.get_joint(char_info.bvh_map[j]).xform_from_parent_joint
            # else:
            #     raise NotImplementedError
            T = ref_pose.get_transform(char_info.bvh_map[j], local=True)
            bvh_map_indices.append(char_info.bvh_map[j])
            Ts_base.append(T)
            
            if joint_type == self._pb_client.JOINT_SPHERICAL:
                As.append(action[dof_cnt:dof_cnt+3])
                dof_cnt += 3
            elif joint_type == self._pb_client.JOINT_REVOLUTE:
                axis = agent.get_joint_axis(j)
                angle = action[dof_cnt:dof_cnt+1]
                As.append(axis*angle)
                dof_cnt += 1
            else:
                raise NotImplementedError
        '''
        Compute new xforms
        '''
        Rs_base, ps_base = conversions.T2Rp(np.array(Ts_base))
        Rs = conversions.A2R(np.array(As))
        '''
        Correct specific joint rotations if necessary
        '''
        if self._joint_correction_info[idx]:
            for idx, axis in self._joint_correction_info[idx]:
                Q_target = conversions.R2Q(Rs[idx])
                Q_closest, _ = quaternion.Q_closest(
                    Q_target, 
                    np.array([0.0, 0.0, 0.0, 1.0]),
                    axis,
                    )
                Rs[idx] = conversions.Q2R(Q_closest)
        '''
        Apply the new xforms
        '''
        Rs_new = np.matmul(Rs_base, Rs)
        Ts_new = conversions.Rp2T(Rs_new, ps_base)
        for idx, j in enumerate(bvh_map_indices):
            ref_pose.set_transform(j, Ts_new[idx], do_ortho_norm=False, local=True)

        return ref_pose

    def compute_init_pose_vel(self, info):
        '''
        This compute initial poses and velocities for all agents.
        The returned poses and velocites will be the initial pose and
        velocities of the simulated agent.
        '''
        init_poses, init_vels = [], []
        for i in range(self._num_agent):
            cur_pose = self._base_motion[i].get_pose_by_frame(0)
            cur_vel = self._base_motion[i].get_velocity_by_frame(0)
            if info.get('add_noise'):
                cur_pose, cur_vel = self._base_env.add_noise_to_pose_vel(
                    self._sim_agent[i], cur_pose, cur_vel)
            init_poses.append(cur_pose)
            init_vels.append(cur_vel)
        return init_poses, init_vels

    def callback_reset_prev(self, info):
        '''
        This is called right before the main reset fn. is called.
        '''
        return

    def callback_reset_after(self, info):
        '''
        This is called right after the main reset fn. is called.
        '''
        return
    
    def reset(self, info):
        
        self.callback_reset_prev(info)

        self._target_pose = [None for i in range(self._num_agent)]
        self._init_poses, self._init_vels = self.compute_init_pose_vel(info)

        self._base_env.reset(time=0.0,
                             poses=self._init_poses, 
                             vels=self._init_vels)
        
        self._end_of_episode = False
        self._end_of_episode_reason = []

        self._end_of_episode_intermediate = False
        self._end_of_episode_reason_intermediate = []
        self._time_elapsed_after_end_of_episode = 0.0

        if "low_reward" in self._early_term_choices:
            for i in range(self._num_agent):
                self._rew_queue[i].clear()
                for j in range(self._rew_queue[i].maxlen):
                    self._rew_queue[i].append(1.0)

        if "high_reward_sum" in self._early_term_choices:
            self._rew_sum = np.zeros(self._num_agent)

        self.callback_reset_after(info)

        self._rew_data_prev = \
            [self.reward_data(i) for i in range(self._num_agent)]

    def callback_step_prev(self, actions, infos):
        return

    def callback_step_after(self, actions, infos):
        return

    def callback_step_end(self, actions, infos):
        return

    def print_log_in_step(self):
        if self._verbose and self._end_of_episode:
            print('=================EOE=================')
            print('Reason:', self._end_of_episode_reason)
            print('(time_elapsed:%02f) (time_after_eoe: %02f) (eoe_margin: %02f)'\
                %(self.get_elapsed_time(),
                  self._time_elapsed_after_end_of_episode,
                  self._eoe_margin))
            print('=====================================')
    
    def step(self, actions):

        rews, infos = [], [{} for i in range(self._num_agent)]

        if profile:
            print('----------------------------------')
            tc.begin()

        self.callback_step_prev(actions, infos)

        if profile:
            print('callback_step_prev:', tc.get_time())

        target_pose = self._num_agent * [None]
        base_residual_linear_force = self._num_agent * [None]
        base_residual_angular_force = self._num_agent * [None]

        action_dict_list = []

        for i in range(self._num_agent):
            action_dict = {
                "pose": None,
                "vel": None,
                "torque": None,
                "base_residual_linear_force": None,
                "base_residual_angular_force": None,
                "base_residual_linear_force_frame": None,
                "base_residual_angular_force_frame": None,
            }

            cnt = 0

            if self._use_base_residual_linear_force:
                act_space = self._action_space[i]['base_residual_linear_force']
                action_dict["base_residual_linear_force"] = \
                    act_space.norm_to_real(actions[i][cnt:cnt+3])
                action_dict["base_residual_linear_force_frame"] = \
                    self._base_residual_linear_force_frame
                cnt += 3

            if self._use_base_residual_angular_force:
                act_space = self._action_space[i]['base_residual_angular_force']
                action_dict["base_residual_angular_force"] = \
                    act_space.norm_to_real(actions[i][cnt:cnt+3])
                action_dict["base_residual_angular_force_frame"] = \
                    self._base_residual_angular_force_frame
                cnt += 3

            if self._sim_agent[i]._actuation==sim_agent.SimAgent.Actuation.NONE:
                pass
            elif self._sim_agent[i]._actuation==sim_agent.SimAgent.Actuation.TQ:
                act_space = self._action_space[i]['torque']                
                action_dict["torque"] = act_space.norm_to_real(actions[i][cnt:])
            else:
                act_space = self._action_space[i]['target_pose']                
                action_dict["pose"] = self.compute_target_pose(
                    i, 
                    act_space.norm_to_real(actions[i][cnt:]))
                self._target_pose[i] = action_dict["pose"]

            action_dict_list.append(action_dict)

        if profile:
            print('reward_data_prev_collect:', tc.get_time())

        if profile:
            print('compute_target_pose:', tc.get_time())
        
        ''' Update simulation '''
        self._base_env.step(action_dict_list)
            # target_poses=target_pose,
            # base_residual_linear_forces=base_residual_linear_force,
            # base_residual_angular_forces=base_residual_angular_force,
            # base_residual_linear_force_frame=self._base_residual_linear_force_frame,
            # base_residual_angular_force_frame=self._base_residual_angular_force_frame,
            # )

        if profile:
            print('base_env_step:', tc.get_time())

        self.callback_step_after(actions, infos)

        if profile:
            print('callback_step_after:', tc.get_time())

        ''' Collect data for reward computation after the current step'''
        rew_data_next = [self.reward_data(i) for i in range(self._num_agent)]

        if profile:
            print('reward_data_next_collect:', tc.get_time())

        ''' 
        Check conditions for end-of-episode. 
        If 'eoe_margin' is larger than zero, the environment will continue for some time.
        '''
        
        if not self._end_of_episode_intermediate:
            eoe_reason = self.inspect_end_of_episode()

            self._end_of_episode_intermediate = len(eoe_reason) > 0
            self._end_of_episode_reason_intermediate = eoe_reason

        if self._end_of_episode_intermediate:
            self._time_elapsed_after_end_of_episode += self._dt_con
            if self._time_elapsed_after_end_of_episode >= self._eoe_margin:
                self._end_of_episode = True
                self._end_of_episode_reason = self._end_of_episode_reason_intermediate

        if profile:
            print('inspect_end_of_episode:', tc.get_time())

        ''' Compute rewards '''
        
        for i in range(self._num_agent):
            r, rd = self.reward(i, self._rew_data_prev, rew_data_next, actions)
            rews.append(r)
            info = {
                'eoe_reason': self._end_of_episode_reason,
                'rew_info': rd,
                'learning_info': self._learning_info
            }
            infos[i].update(info)
            if "low_reward" in self._early_term_choices:
                self._rew_queue[i].append(r)
            if "high_reward_sum" in self._early_term_choices:
                self._rew_sum[i] += r
        self._rew_data_prev = rew_data_next

        if profile:
            print('compute_reward:', tc.get_time())

        if profile:
            print('----------------------------------')

        self.callback_step_end(actions, infos)

        if profile:
            print('callback_step_end:', tc.get_time())

        self.print_log_in_step()
        
        return rews, infos

    def state(self, idx, state_choices=None):
        state = []

        if state_choices is None:
            state_choices = self._state_choices
        for key in state_choices:
            state.append(self.get_state_by_key(idx, key))

        return np.hstack(state)

    @abstractmethod
    def get_state_by_key(self, idx, key):
        '''
        This returns state that corresponds to a given key
        '''
        raise NotImplementedError    

    def state_body(self, idx):
        '''
        This returns proprioceptive state of an agent as a numpy array
        '''
        raise NotImplementedError

    def _state_body(self, 
                    idx, 
                    agent,
                    type=None,
                    return_stacked=True):
        if type is None:
            type = self._state_body_type
        
        if type == "facing":
            T_ref=agent.get_facing_transform(self.get_ground_height(idx))
            include_com=True
            include_link_p=True
            include_link_Q=True
            include_link_v=True
            include_link_w=True
            include_link_R6=False
            include_root_as_link=True
            include_root_height=False
            include_root_up_dir=False
        elif type == "facing_R6_h":
            T_ref=agent.get_facing_transform(self.get_ground_height(idx))
            include_com=False
            include_link_p=True
            include_link_Q=False
            include_link_v=True
            include_link_w=True
            include_link_R6=True
            include_root_as_link=False
            include_root_height=True
            include_root_up_dir=False
        elif type == "root_R6_h":
            T_ref=agent.get_root_transform()
            include_com=False
            include_link_p=True
            include_link_Q=False
            include_link_v=True
            include_link_w=True
            include_link_R6=True
            include_root_as_link=False
            include_root_height=True
            include_root_up_dir=False
        elif type == "root_R6_h_vup":
            T_ref=agent.get_root_transform()
            include_com=False
            include_link_p=True
            include_link_Q=False
            include_link_v=True
            include_link_w=True
            include_link_R6=True
            include_root_as_link=False
            include_root_height=True
            include_root_up_dir=True
        else:
            raise NotImplementedError

        return self._state_body_raw(
            idx,
            agent,
            T_ref=T_ref, 
            include_com=include_com, 
            include_link_p=include_link_p, 
            include_link_Q=include_link_Q, 
            include_link_v=include_link_v, 
            include_link_w=include_link_w, 
            include_link_R6=include_link_R6,
            include_root_as_link=include_root_as_link,
            include_root_height=include_root_height,
            include_root_up_dir=include_root_up_dir,
            return_stacked=return_stacked,
        )

    def _state_body_raw(
        self, 
        idx,
        agent, 
        T_ref,
        include_com, 
        include_link_p, 
        include_link_Q, 
        include_link_v, 
        include_link_w, 
        include_link_R6,
        include_root_as_link,
        include_root_height,
        include_root_up_dir,
        return_stacked):
        R_ref, p_ref = conversions.T2Rp(T_ref)
        R_ref_inv = R_ref.transpose()

        link_states = []
        if include_root_as_link:
            link_states.append(agent.get_root_state())
        ps, Qs, vs, ws = agent.get_link_states()
        for j in agent._joint_indices:
            link_states.append((ps[j], Qs[j], vs[j], ws[j]))

        state = []
        for i, s in enumerate(link_states):
            p, Q, v, w = s[0], s[1], s[2], s[3]
            if include_link_p:
                p_rel = np.dot(R_ref_inv, p - p_ref)
                state.append(p_rel) # relative position w.r.t. the reference frame
            if include_link_Q:
                Q_rel = conversions.R2Q(np.dot(R_ref_inv, conversions.Q2R(Q)))
                Q_rel = quaternion.Q_op(Q_rel, op=["normalize", "halfspace"])
                state.append(Q_rel) # relative rotation w.r.t. the reference frame
            if include_link_v:
                v_rel = np.dot(R_ref_inv, v)
                state.append(v_rel) # relative linear vel w.r.t. the reference frame
            if include_link_w:
                w_rel = np.dot(R_ref_inv, w)
                state.append(w_rel) # relative angular vel w.r.t. the reference frame
            if include_link_R6:
                R = conversions.Q2R(Q)
                r0, r1 = R[:,0], R[:,1]
                state.append(np.dot(R_ref_inv, r0))
                state.append(np.dot(R_ref_inv, r1))
            if include_com:
                if i==0:
                    p_com = agent._link_masses[i] * p
                    v_com = agent._link_masses[i] * v
                else:
                    p_com += agent._link_masses[i] * p
                    v_com += agent._link_masses[i] * v

        if include_com:
            p_com /= agent._link_total_mass
            v_com /= agent._link_total_mass
            state.append(np.dot(R_ref_inv, p_com - p_ref))
            state.append(np.dot(R_ref_inv, v_com))

        if include_root_height:
            state.append(
                agent.get_root_height_from_ground(self.get_ground_height(idx)))

        if include_root_up_dir:
            R, p = conversions.T2Rp(agent.get_root_transform())
            v_up = np.dot(R, agent._char_info.v_up)
            state.append(v_up)
        
        if return_stacked:
            return np.hstack(state)
        else:
            return state

    def state_task(self, idx):
        '''
        This returns a task-specifit state (numpy array)
        '''     
        raise NotImplementedError

    @abstractmethod
    def reward_data(self, idx):
        '''
        This returns a dictionary that includes data to compute reward value
        '''
        raise NotImplementedError

    @abstractmethod
    def reward_max(self):
        '''
        This returns a maximum reward value
        '''
        raise NotImplementedError

    @abstractmethod
    def reward_min(self):
        '''
        This returns a minimum reward value
        '''
        raise NotImplementedError

    def return_max(self, gamma):
        '''
        This returns a maximum return (sum of rewards)
        '''
        assert gamma < 1.0
        return self.reward_max() / (1.0 - gamma)

    def return_min(self, gamma):
        '''
        This returns a minimum return (sum of rewards)
        '''
        assert gamma < 1.0
        return self.reward_min() / (1.0 - gamma)

    @abstractmethod
    def get_task_error(self, idx, data_prev, data_next, actions):
        '''
        This computes a task-specific error and 
        returns a dictionary that includes those errors
        '''
        raise NotImplementedError

    def reward(self, idx, data_prev, data_next, actions):
        '''
        This returns a reward, and a dictionary
        '''   
        
        error = self.get_task_error(idx, data_prev, data_next, actions)

        rew_fn_def = self._reward_fn_def[self._reward_fn_map[idx]]
        rew, rew_info = self.compute_reward(error, rew_fn_def)

        return float(rew), rew_info

    def get_reward_fn_subterms_in_use(self, fn_def):
        rew_names = set()
        op = fn_def['op']

        if op in ['add', 'mul', 'min', 'max']:
            for child in fn_def['child_nodes']:
                rew_names = rew_names.union(self.get_reward_fn_subterms_in_use(child))
        elif op == 'leaf' or op == 'constant':
            rew_names.add(fn_def['name'])
        else:
            raise NotImplementedError

        return rew_names

    def pretty_print_rew_info(self, rew_info, prefix=str()):
        print("%s > name:   %s"%(prefix, rew_info['name']))
        print("%s   value:  %s"%(prefix, rew_info['value']))
        print("%s   weight: %s"%(prefix, rew_info['weight']))
        print("%s   op: %s"%(prefix, rew_info['op']))
        for child in rew_info["child_nodes"]:
            self.pretty_print_rew_info(child, prefix+"\t")

    def compute_reward(self, error, fn_def):
        ''' 
        This computes a reward by using 
        task-specific errors and the reward definition tree
        '''
        # op = fn_def['op']
        # n = fn_def['name'] if 'name' in fn_def.keys() else 'noname'
        # w = fn_def['weight'] if 'weight' in fn_def.keys() else 1.0
        op = fn_def.get('op', 'leaf')
        n = fn_def.get('name', 'noname')
        w = fn_def.get('weight', 1.0)
        v = fn_def.get('value', 0.0)

        rew_info = {'name': n, 'value': v, 'op': op, 'weight': w, 'child_nodes': []}

        if op == 'add':
            rew = 0.0
            for child in fn_def['child_nodes']:
                r, r_info = self.compute_reward(error, child)
                rew += r
                rew_info['child_nodes'].append(r_info)
        elif op == 'mul':
            rew = 1.0
            for child in fn_def['child_nodes']:
                r, r_info = self.compute_reward(error, child)
                rew *= r
                rew_info['child_nodes'].append(r_info)
        elif op == 'min':
            for child in fn_def['child_nodes']:
                r, r_info = self.compute_reward(error, child)
                rew_info['child_nodes'].append(r_info)
            rew = np.min(r)
        elif op == 'max':
            for child in fn_def['child_nodes']:
                r, r_info = self.compute_reward(error, child)
                rew_info['child_nodes'].append(r_info)
            rew = np.max(r)
        elif op == 'constant':
            rew = v
        elif op == 'leaf':
            if 'kernel' in fn_def.keys():
                kernel = fn_def['kernel']
            else:
                kernel = None

            if 'weight_schedule' in fn_def.keys():
                timesteps_total = self._learning_info['timesteps_total']
                w *= math.lerp_from_paired_list(
                    timesteps_total, fn_def['weight_schedule'])

            if isinstance(error[n], float) or isinstance(error[n], list):
                e = np.array([error[n]])
            else:
                e = error[n]

            assert isinstance(e, np.ndarray), \
                "errors should be return as ndarray or float"
            
            if kernel is None or kernel['type'] == "none":
                rew = e
            elif kernel['type'] == "gaussian":
                rew = np.exp(-kernel['scale']*e)
            elif kernel['type'] == "quadratic":
                v = kernel['scale']*e
                rew = np.dot(v, v)
            else:
                raise NotImplementedError
            rew *= w
        else:
            raise NotImplementedError

        rew_info['value'] = rew

        return rew, rew_info

    def inspect_end_of_episode(self):
        '''
        This checks whether END-OF-EPISODE events happen and returns 
        a list that includes reasons
        '''
        eoe_reason = []
        for idx in range(self._num_agent):
            name = self._sim_agent[idx].get_name()
            ''' Terminate when the agents falldown '''
            if "falldown" in self._early_term_choices:
                check = self._base_env.check_falldown(idx=idx)
                if check: eoe_reason.append('[%s]:falldown'%name)
            ''' Terminate when the simulation diverges '''
            if "sim_div" in self._early_term_choices:
                check = self._base_env.check_sim_divergence(self._sim_agent[idx])
                if check: eoe_reason.append('[%s]:sim_div'%name)
            ''' Terminate when the given time elapses '''
            if "sim_window" in self._early_term_choices:
                check = self.get_elapsed_time() > self._sim_window_time
                if check: eoe_reason.append('[%s]:sim_window'%name)
            ''' Terminate when the average reward is lower than a specified value '''
            if "low_reward" in self._early_term_choices:
                check = np.mean(list(self._rew_queue[idx])) < self._et_low_reward_thres
                if check: eoe_reason.append('[%s]:low_reward'%name)
            ''' Terminate when the cumulative reward is larger than a specified value '''
            if "high_reward_sum" in self._early_term_choices:
                check = self._rew_sum >= self._et_high_rew_sum_thres
                if check: eoe_reason.append('[%s]:high_reward_sum'%name)
            ''' Terminate when the agent is out of map '''
            if "out_of_ground" in self._early_term_choices:
                check = self._base_env.check_out_of_ground(self._sim_agent[idx])
                if check: eoe_reason.append('[%s]:out_of_ground'%name)
        return eoe_reason

    def get_ground_height_at(self, p):
        return self._base_env.get_ground_height_at(p)

    def get_ground_height(self, idx):
        return self._base_env.get_ground_height(idx)

    def get_ground_height_all(self):
        return self._base_env.get_ground_height_all(idx)

    def get_elapsed_time(self):
        '''
        This returns the elpased time after the environment was reset
        '''
        return self._base_env._elapsed_time

    def set_elapsed_time(self, time):
        self._base_env._elapsed_time = time

    def check_falldown(self, idx):
        return self._base_env.check_falldown(idx)

    def get_render_data(self, idx):
        return self._base_env.get_render_data(self._sim_agent[idx])

    def render(self, rm):
        colors = rm.COLORS_FOR_AGENTS

        rm.gl.glEnable(rm.gl.GL_LIGHTING)
        rm.gl.glEnable(rm.gl.GL_BLEND)
        rm.gl.glBlendFunc(rm.gl.GL_SRC_ALPHA, rm.gl.GL_ONE_MINUS_SRC_ALPHA)

        self._base_env.render(rm)

        if rm.flag['target_pose']:
            for i in range(self._num_agent):
                if self._target_pose[i] is None: continue
                agent = self._kin_agent[i]
                agent_state = agent.save_states()
                agent.set_pose(self._target_pose[i])
                rm.gl.glPushAttrib(rm.gl.GL_LIGHTING|rm.gl.GL_DEPTH_TEST|rm.gl.GL_BLEND)
                rm.bullet_render.render_model(self._pb_client, 
                                              agent._body_id,
                                              draw_link=True,
                                              draw_link_info=False,
                                              draw_joint=rm.flag['joint'],
                                              draw_joint_geom=False, 
                                              ee_indices=agent._char_info.end_effector_indices,
                                              color=[colors[i][0], colors[i][1], colors[i][2], 0.5])
                rm.gl.glPopAttrib()
                agent.restore_states(agent_state)

        if rm.flag['kin_model']:
            for i in range(self._num_agent):
                agent = self._kin_agent[i]                
                rm.gl.glPushAttrib(rm.gl.GL_LIGHTING|rm.gl.GL_DEPTH_TEST|rm.gl.GL_BLEND)
                rm.bullet_render.render_model(self._pb_client, 
                                              agent._body_id,
                                              draw_link=True,
                                              draw_link_info=False,
                                              draw_joint=rm.flag['joint'],
                                              draw_joint_geom=False, 
                                              ee_indices=agent._char_info.end_effector_indices,
                                              color=[1-colors[i][0], 1-colors[i][1], 1-colors[i][2], 0.3])
                if rm.flag['com_vel']:
                    p, Q, v, w = agent.get_root_state()
                    p, v = agent.get_com_and_com_vel()
                    rm.gl_render.render_arrow(p, p+v, D=0.01, color=[0.5, 0.5, 0.5, 1])
                rm.gl.glPopAttrib()
