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

import gym
from gym.spaces import Box

from envs import env_humanoid_imitation as my_env
import env_renderer as er
import render_module as rm

import pickle

import rllib_model_torch as policy_model
from collections import deque

from ray.rllib.utils import try_import_torch
torch, nn = try_import_torch()

from fairmotion.core.motion import Pose, Motion
from fairmotion.core.velocity import MotionWithVelocity
from fairmotion.data import bvh
from fairmotion.ops import conversions

import utils
    
def gen_state_action_pairs(trainer, env):
    model = trainer.get_policy().model
    base_env = env.base_env
    print('----------------------------')
    print('Extracting State-Action Pairs')
    print('----------------------------')
    ''' Read a directory for saving images and try to create it '''
    output_dir = input("Enter output dir: ")
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError:
        print("Invalid Subdirectory")
        return
    iter_per_episode = 10
    window_size = 1000
    stride = 1000
    state_type = "facing_R6_h"
    exp_std = 0.05
    model.set_exploration_std(exp_std)
    def state_body_custom(type):
        return base_env.state_body(
            idx=0, 
            agent="sim", 
            type=type, 
            return_stacked=True)
    data = {
        'iter_per_episode': iter_per_episode,
        'dim_state': base_env.dim_state(0),
        'dim_state_body': len(state_body_custom(state_type)),
        'dim_state_task': base_env.dim_state_task(0),
        'dim_action': base_env.dim_action(0),
        'episodes': [],
        'exp_std': exp_std,
    }
    for i in range(len(base_env._ref_motion_all[0])):
        for j in range(iter_per_episode):
            cnt_per_trial = 0
            time_start = -window_size + stride
            falldown_cnt = 0
            while True:
                print("\rtime_start: ", time_start, end='')
                episode = {
                    'time': [],
                    'state': [],
                    'action': [],
                    'action_gt': [],
                    'reward': [],
                    'state_body': [],
                    'state_task': [],
                }
                env.reset({
                    'ref_motion_id': [i], 
                    'start_time': np.array([max(0.0, time_start)])}
                )
                time_elapsed = max(0.0, time_start) - time_start
                falldown = False
                cnt_per_window = 0
                while True:
                    s1 = env.state()
                    s1_body = state_body_custom(state_type)
                    s1_task = base_env.state_task(0)
                    a = trainer.compute_single_action(s1, explore=True)
                    a_gt = trainer.compute_single_action(s1, explore=False)
                    s2, rew, eoe, info = env.step(a)
                    t = base_env.get_current_time()
                    time_elapsed += base_env._dt_con
                    episode['time'].append(t)
                    episode['state'].append(s1)
                    episode['action'].append(a)
                    episode['action_gt'].append(a_gt)
                    episode['reward'].append(rew)
                    episode['state_body'].append(s1_body)
                    episode['state_task'].append(s1_task)
                    cnt_per_window += 1
                    ''' 
                    The policy output might not be ideal 
                    when no future reference motion exists.
                    '''
                    if base_env._ref_motion[0].length()-base_env.get_current_time() \
                        <= base_env._sensor_lookahead[-1]:
                        break
                    if time_elapsed >= window_size:
                        break
                    if base_env._end_of_episode:
                        falldown = True
                        break
                ''' Include only successful (not falling) episodes '''
                if not falldown:
                    data['episodes'].append(episode)    
                    time_start += stride
                    cnt_per_trial += cnt_per_window
                    ''' End if not enough frames remain '''
                    if base_env._ref_motion[0].length() < time_start + stride:
                        break
                else:
                    falldown_cnt += 1
                    if falldown_cnt >= 10:
                        print("****** TOO MANY FALLDOWN, Skipping ******")
                        time_start += stride
                    else:
                        print("****** FALLDOWN, Retrying ****** %d"%falldown_cnt)
            print('\n%d pairs were created in %d-th trial of episode%d'%(cnt_per_trial, j, i))
    output_file = os.path.join(
        output_dir, 
        "data_iter=%d,winsize=%.2f,stride=%.2f,state_type=%s,exp_std=%.2f.pkl"%(iter_per_episode, window_size, stride, state_type, exp_std))
    with open(output_file, "wb") as file:
        pickle.dump(data, file)
        print("Saved:", file)

class HumanoidImitation(gym.Env):
    def __init__(self, env_config):
        self.base_env = my_env.Env(env_config)
        assert self.base_env._num_agent == 1
        
        ob_scale = 1000.0
        dim_state = self.base_env.dim_state(0)
        dim_state_body = self.base_env.dim_state_body(0)
        dim_state_task = self.base_env.dim_state_task(0)
        dim_action = self.base_env.dim_action(0)
        action_range_min, action_range_max = self.base_env.action_range(0)

        self.observation_space = \
            Box(-ob_scale * np.ones(dim_state),
                ob_scale * np.ones(dim_state),
                dtype=np.float64)
        self.observation_space_body = \
            Box(-ob_scale * np.ones(dim_state_body),
                ob_scale * np.ones(dim_state_body),
                dtype=np.float64)
        self.observation_space_task = \
            Box(-ob_scale * np.ones(dim_state_task),
                ob_scale * np.ones(dim_state_task),
                dtype=np.float64)
        self.action_space = \
            Box(action_range_min,
                action_range_max,
                dtype=np.float64)

    def state(self):
        return self.base_env.state(idx=0)

    def reset(self, info={}):
        if not self.base_env._initialized:
            self.base_env.create()
        self.base_env.reset(info)
        return self.base_env.state(idx=0)

    def step(self, action):
        rew, info = self.base_env.step([action])
        obs = self.state()
        eoe = self.base_env._end_of_episode
        if self.base_env._verbose:
            print(rew)
        return obs, rew[0], eoe, info[0]

class EnvRunner(object):
    def __init__(self, trainers, env, config):
        self.trainer = trainers[0]
        self.env = env
        self.config = config
        self.explore = True
        '''
        full: Evaluate the entire model
        pass_through: Evaluate the motor_decoder only
        '''
        self.policy_eval_method = 0
        self.policy_eval_method_list = ['full', 'pass_through']
        self.reset()
    def reset(self, info={}):
        self.env.reset(info)
        self.policy_hidden_state = self.trainer.get_policy().get_initial_state()
        self.data = {
            'z_task': deque(maxlen=30),
            'gate_fn_w': deque(maxlen=30),
            'joint_data': list(),
            'link_data': list(),
        }
    def one_step(self, explore=None):
        if explore is None:
            explore = self.explore
        # Run the environment
        s1 = self.env.state()
        policy = self.trainer.get_policy()
        method = self.policy_eval_method_list[self.policy_eval_method]
        if  method == 'full':
            if policy.is_recurrent():
                action, state_out, extra_fetches = \
                    self.trainer.compute_single_action(
                        s1, 
                        state=self.policy_hidden_state,
                        explore=explore)
                self.policy_hidden_state = state_out
            else:
                action = self.trainer.compute_single_action(
                    s1, 
                    explore=explore)
        elif method == 'pass_through' and \
            isinstance(policy.model, (policy_model.PhysicsVAE)):
            prior_type = policy.model._latent_prior_type
            if prior_type == 'normal_zero_mean_one_std':
                z_task = np.random.normal(
                    size=policy.model._task_encoder_output_dim)
            elif prior_type == 'normal_state_mean_one_std':
                _, _, _ = policy.model.forward_encoder(
                    obs=torch.Tensor([s1]),
                    state=None,
                    seq_lens=None,
                    state_cnt=None,
                )
                mean = policy.model._cur_latent_prior_mu[0].detach().numpy().copy()
                logstd = policy.model._cur_latent_prior_logvar[0].detach().numpy().copy()
                z_task = np.random.normal(loc=mean, scale=np.exp(logstd))
            else:
                raise NotImplementedError
            logits, _ = policy.model.forward_decoder(
                z_body=torch.Tensor([self.env.base_env.state_body(0)]),
                z_task=torch.Tensor([z_task]),
                state=None,
                seq_lens=None,
                state_cnt=None,
            )
            mean = logits[0][:self.env.action_space.shape[0]].detach().numpy().copy()
            if explore:
                logstd = logits[0][self.env.action_space.shape[0]:].detach().numpy().copy()
                action = np.random.normal(loc=mean, scale=np.exp(logstd))
            else:
                action = mean
        else:
            raise NotImplementedError
            
        # Step forward
        s2, rew, eoe, info = self.env.step(action)        
        return s2, rew, eoe, info

class EnvRenderer(er.EnvRenderer):
    def __init__(self, trainers, config, **kwargs):
        from fairmotion.viz.utils import TimeChecker
        # kwargs['renderer'] = 'bullet_native'
        super().__init__(**kwargs)
        self.env_runner = EnvRunner(
            trainers=trainers, 
            env=self.env,
            config=config)
        self.time_checker_auto_play = TimeChecker()
        self.bgcolor=[1.0, 1.0, 1.0, 1.0]
        self.cam_params = deque(maxlen=30)
        self.cam_param_offset = None
        self.replay = False
        self.replay_cnt = 0
        self.replay_data = {}
        self.replay_render_interval = 15
        self.replay_render_alpha = 0.5
        self.reset()
    def use_default_ground(self):
        return True
    def get_v_up_env_str(self):
        return self.env.base_env._v_up_str
    def get_ground(self):
        return self.env.base_env._ground
    def get_pb_client(self):
        return self.env.base_env._pb_client
    def reset(self, info={}):
        self.replay_cnt = 0
        if self.replay:
            self.set_pose()
        else:
            self.env_runner.reset()
            self.replay_data = {
                'motion': Motion(
                    skel=self.env.base_env._base_motion[0].skel,
                    fps=self.env.base_env._base_motion[0].fps),
                'joint_data': list(),
                'link_data': list(),
                'others': {},
            }
        self.cam_params.clear()
        param = self._get_cam_parameters()
        for i in range(self.cam_params.maxlen):
            self.cam_params.append(param)
    def collect_replay_data(self):
        sim_agent = self.env.base_env._sim_agent[0]
        motion = self.replay_data['motion']
        motion.add_one_frame(
            sim_agent.get_pose(motion.skel, apply_height_offset=False).data)
        joint_data, link_data = self.env.base_env.get_render_data(0)
        self.replay_data['joint_data'].append(joint_data)
        self.replay_data['link_data'].append(link_data)
    def set_pose(self):
        if self.replay_data['motion'].num_frames() == 0: 
            print('>> No replay data is stored')
            return
        motion = self.replay_data['motion']
        pose = motion.get_pose_by_frame(self.replay_cnt)
        self.env.base_env._sim_agent[0].set_pose(pose)
    def one_step(self):
        self.cam_params.append(self._get_cam_parameters())
        if self.replay:
            self.set_pose()
            self.replay_cnt = \
                min(self.replay_data['motion'].num_frames()-1, self.replay_cnt+1)
        else:
            s2, rew, eoe, info = self.env_runner.one_step()
            self.collect_replay_data()
            return s2, rew, eoe, info
    def extra_render_callback(self):
        if self.rm.get_flag('custom4'):
            '''
            Render all saved intermediate postures
            '''
            if self.replay_data['motion'].num_frames() == 0: 
                return
            motion = self.replay_data['motion']
            pose = motion.get_pose_by_frame(self.replay_cnt)
            self.env.base_env._sim_agent[0].set_pose(pose)
            motion = self.replay_data['motion']
            color = self.rm.COLOR_AGENT.copy()
            color[3] = self.replay_render_alpha
            for i in range(0, motion.num_frames(), self.replay_render_interval):
                if motion.num_frames()-i < self.replay_render_interval:
                    break
                pose = motion.get_pose_by_frame(i)
                agent = self.env.base_env._sim_agent[0]
                agent.set_pose(pose)
                self.rm.bullet_render.render_model(
                    agent._pb_client, 
                    agent._body_id,
                    draw_link=True, 
                    draw_link_info=False, 
                    draw_joint=self.rm.flag['joint'], 
                    draw_joint_geom=False, 
                    ee_indices=agent._char_info.end_effector_indices, 
                    link_info_scale=self.rm.LINK_INFO_SCALE,
                    link_info_line_width=self.rm.LINK_INFO_LINE_WIDTH,
                    link_info_num_slice=self.rm.LINK_INFO_NUM_SLICE,
                    color=color)
        self.env.base_env.render(self.rm)

    def extra_overlay_callback(self):
        w, h = self.window_size
        env = self.env.base_env
        font = self.rm.glut.GLUT_BITMAP_9_BY_15
        ref_motion_name = env._ref_motion_file_names[0][env._ref_motion_idx[0]]
        self.rm.gl_render.render_text(
            "File: %s"%ref_motion_name, pos=[0.05*w, 0.05*h], font=font)

    def extra_idle_callback(self):
        time_elapsed = self.time_checker_auto_play.get_time(restart=False)
        if self.rm.flag['auto_play'] and time_elapsed >= self.env.base_env._dt_con:
            self.time_checker_auto_play.begin()
            self.env_runner.one_step()
    def extra_keyboard_callback(self, key):
        if key == b'r':
            print("Reset w/o replay")
            self.reset()
        elif key == b' ':
            self.time_checker_auto_play.begin()
            self.one_step()
        elif key == b'e':
            self.env_runner.explore = not self.env_runner.explore
            print('Exploration:', self.env_runner.explore)
        elif key == b'E':
            model = self.env_runner.trainer.get_policy().model
            exp_std = utils.get_float_from_input("Exploration Std")
            assert exp_std >= 0.0
            model.set_exploration_std(exp_std)
        elif key == b'l':
            file = input("Enter Model Weight File:")
            if os.path.isfile(file):
                model = self.env_runner.trainer.get_policy().model
                model.load_weights(file)
                print(file)
        elif key == b':':
            file = input("Enter Motor Decoder Weight File:")
            if os.path.isfile(file):
                model = self.env_runner.trainer.get_policy().model
                model.load_weights_motor_decoder(file)
                print(file)
        elif key == b'L':
            gen_state_action_pairs(self.env_runner.trainer, self.env)
        elif key == b'q':
            self.env_runner.policy_eval_method = \
                (self.env_runner.policy_eval_method+1)%len(self.env_runner.policy_eval_method_list)
            print("policy_eval_method:", self.env_runner.policy_eval_method_list[self.env_runner.policy_eval_method])

    def _get_cam_parameters(self, apply_offset=True):
        param = {
            "origin": None, 
            "pos": None, 
            "dist": None,
            "translate": None,
        }
        
        agent = self.env.base_env._sim_agent[0]
        h = self.env.base_env.get_ground_height(0)
        d_face, p_face = agent.get_facing_direction_position(h)
        origin = p_face + agent._char_info.v_up_env

        if self.rm.get_flag("follow_cam") == "pos+rot":
            pos = p_face + 2 * (agent._char_info.v_up_env - d_face)
        else:
            pos = self.cam_cur.pos + (origin - self.cam_cur.origin)
        
        if apply_offset and self.cam_param_offset is not None:
            if self.rm.get_flag("follow_cam") == "pos+rot":
                R_face, _ = conversions.T2Rp(agent.get_facing_transform(h))
                pos += np.dot(R_face, self.cam_param_offset[1])
                origin += np.dot(R_face, self.cam_param_offset[0])
        
        param["origin"] = origin
        param["pos"] = pos
        
        return param
    def get_cam_parameters(self, use_buffer=True):
        if use_buffer:
            param = {
                "origin": np.mean([p["origin"] for p in self.cam_params], axis=0), 
                "pos": np.mean([p["pos"] for p in self.cam_params], axis=0), 
            }
        else:
            param = self._get_cam_parameters()
        return param
    def get_elapsed_time(self):
        return self.env.base_env.get_elapsed_time()

def default_cam(env):
    agent = env.base_env._sim_agent[0]
    # R, p = conversions.T2Rp(agent.get_facing_transform(0.0))    
    v_up_env = agent._char_info.v_up_env
    v_up = agent._char_info.v_up
    v_face = agent._char_info.v_face
    origin = np.zeros(3)
    return rm.camera.Camera(
        pos=3*(v_up+v_face),
        origin=origin, 
        vup=v_up_env, 
        fov=60.0)

env_cls = HumanoidImitation

def config_override(spec):
    env = env_cls(spec["config"]["env_config"])

    model_config = copy.deepcopy(spec["config"]["model"])
    if hasattr(env, "observation_space_body"):
        model_config.get("custom_model_config").update({
            "observation_space_body": copy.deepcopy(env.observation_space_body)})
    if hasattr(env, "observation_space_task"):
        model_config.get("custom_model_config").update({
            "observation_space_task": copy.deepcopy(env.observation_space_task)})            
    del env

    config = {
        # "callbacks": {},
        "model": model_config,
    }
    return config
