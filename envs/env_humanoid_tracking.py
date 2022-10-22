# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(directory))

''' 
This forces the environment to use only 1 cpu when running.
This is helpful to launch multiple environment simulatenously.
'''
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import copy

import pybullet as pb
import pybullet_data

from bullet import bullet_client
from bullet import bullet_utils as bu

from fairmotion.ops import conversions
from fairmotion.ops import math
from fairmotion.utils import constants
from fairmotion.utils import utils

import sim_agent
import sim_obstacle
import importlib.util

from PIL import Image, ImageOps

profile = False
if profile:
    from fairmotion.viz.utils import TimeChecker
    tc = TimeChecker()

class Env(object):
    '''
    This environment defines a base environment where the simulated 
    characters exist and they are controlled by tracking controllers
    '''
    def __init__(self, config):
        project_dir                  = config["project_dir"]
        fps_sim                      = config.get("fps_sim", 480)
        fps_con                      = config.get("fps_con", 30)
        solver_num_steps             = config.get("solver_num_steps", 2)
        solver_iter                  = config.get("solver_iter", 10)
        friction_cone                = config.get("friction_cone", True)
        verbose                      = config.get("verbose", False)

        char_info_module             = config["character"]["char_info_module"]
        sim_char_file                = config["character"]["sim_char_file"]

        self._num_agent = len(sim_char_file)

        ref_motion_scale             = \
            config["character"].get("ref_motion_scale", np.ones(self._num_agent))
        ref_motion_height_fix        = \
            config["character"].get("ref_motion_height_fix", np.zeros(self._num_agent))
        physics_param                = config["character"]["physics"]
        contactable_body             = config["character"].get("contactable_body", None)
        
        ground_enable                = config["ground"].get("enable", True)
        ground_friction_coeff        = config["ground"].get("friction_coeff")
        ground_contact_stiffness     = config["ground"].get("contact_stiffness")
        ground_contact_damping       = config["ground"].get("contact_damping")
        ground_urdf_file             = config["ground"].get("urdf_file", None)
        ground_height_measure_offset = config["ground"].get("height_measure_offset", 5.0)
        ground_height_map            = config["ground"].get("height_map", None)
        ground_size                  = config["ground"].get("size", [100, 100, 10])
        ground_falldown_check        = config["ground"].get("falldown_check", "collision")

        self._config = config

        assert self._num_agent > 0
        assert self._num_agent == len(char_info_module)
        assert self._num_agent == len(ref_motion_scale)
        assert self._num_agent == len(ref_motion_height_fix)
        assert self._num_agent == len(physics_param)

        ''' Append project_dir to the given file path '''
        
        for i in range(len(char_info_module)):
            char_info_module[i] = os.path.join(project_dir, char_info_module[i])
            sim_char_file[i]    = os.path.join(project_dir, sim_char_file[i])
        if ground_urdf_file is not None:
            ground_urdf_file = os.path.join(project_dir, ground_urdf_file)
        if ground_height_map is not None:
            ground_height_map = os.path.join(project_dir, ground_height_map)

        self._char_info = []
        for i in range(self._num_agent):
            ''' Load Character Info Moudle '''
            spec = importlib.util.spec_from_file_location(
                "char_info%d"%(i), char_info_module[i])
            char_info = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(char_info)
            self._char_info.append(char_info)
            ''' Modfiy Contactable Body Parts '''
            if contactable_body:
                contact_allow_all = True if 'all' in contactable_body else False
                for joint in list(char_info.contact_allow_map.keys()):
                    char_info.contact_allow_map[joint] = \
                        contact_allow_all or char_info.joint_name[joint] in contactable_body

        self._v_up = self._char_info[0].v_up_env
        self._v_up_str = utils.axis_to_str(self._v_up)

        ''' Define PyBullet Client '''
        self._pb_client = bullet_client.BulletClient(
            connection_mode=pb.DIRECT, options=' --opengl2')
        self._pb_client.setAdditionalSearchPath(pybullet_data.getDataPath())

        self._fps_sim = fps_sim
        self._fps_con = fps_con
        ''' timestep for physics simulation '''
        self._dt_sim = 1.0/fps_sim
        ''' timestep for control of dynamic controller '''
        self._dt_con = 1.0/fps_con

        if fps_sim%fps_con != 0:
            raise Exception('FPS_SIM should be a multiples of FPS_ACT')
        self._num_substep = fps_sim//fps_con

        self._verbose = verbose

        self.setup_agents(
            sim_char_file, 
            self._char_info, 
            ref_motion_scale,
            ref_motion_height_fix, 
            physics_param)

        self._ground = None
        self._ground_height_map = None
        if ground_enable:
            self._use_default_ground = ground_urdf_file is None
            if self._use_default_ground:
                ground_urdf_file = "plane_implicit.urdf"
            self._ground = \
                self.create_ground(
                    ground_urdf_file,
                    ground_friction_coeff,
                    ground_contact_stiffness,
                    ground_contact_damping)
            self._ground_size = ground_size
            self._ground_height_measure_offset = ground_height_measure_offset
            if ground_height_map:
                im = Image.open(ground_height_map)
                im = im.rotate(180)
                # im = ImageOps.flip(im)
                self._ground_height_map = np.asarray(im).astype('float32')/255.0
                self._ground_height_map_size = im.size
            self._ground_falldown_check = ground_falldown_check

        self.setup_others(friction_cone, solver_num_steps, solver_iter)

        ''' Elapsed time after the environment starts '''
        self._elapsed_time = 0.0
        ''' For tracking the length of current episode '''
        self._episode_len = 0.0
        
        ''' Create a Manager for Handling Obstacles '''
        self._obs_manager = sim_obstacle.ObstacleManager(
            self._pb_client, self._dt_con, self._v_up)

        ''' Save the initial pybullet state to clear all thing before calling reset '''
        self._init_state = None
        self.reset()
        self._init_state = self._pb_client.saveState()

    def setup_agents(
        self, 
        sim_char_file, 
        char_info, 
        ref_motion_scale, 
        ref_motion_height_fix,
        physics_param):
        self._pb_client.resetSimulation()
        self._agent = []
        for i in range(self._num_agent):
            keys = physics_param[i].keys()
            if 'friction_lateral' in keys:
                char_info[i].friction_lateral = physics_param[i]['friction_lateral']
            if 'friction_spinning' in keys:
                char_info[i].friction_spinning = physics_param[i]['friction_spinning']
            if 'restitution' in keys:
                char_info[i].restitution = physics_param[i]['restitution']
            self._agent.append(
                sim_agent.SimAgent(
                    name='sim_agent_%d'%(i),
                    pybullet_client=self._pb_client, 
                    model_file=sim_char_file[i], 
                    char_info=char_info[i], 
                    ref_scale=ref_motion_scale[i],
                    ref_height_fix=ref_motion_height_fix[i],
                    self_collision=physics_param[i]['self_collision'],
                    actuation=physics_param[i]['actuation'],
                    kinematic_only=False,
                    verbose=self._verbose))

    def create_ground(
        self, 
        ground_urdf_file,
        friction_coeff=None,
        contact_stiffness=None, 
        contact_damping=None, 
        ):
        ''' Create Plane '''
        if np.allclose(np.array([0.0, 0.0, 1.0]), self._v_up):
            R_plane = constants.eye_R()
        else:
            R_plane = math.R_from_vectors(np.array([0.0, 0.0, 1.0]), self._v_up)
        ground = \
            self._pb_client.loadURDF(
                ground_urdf_file, 
                [0, 0, 0], 
                conversions.R2Q(R_plane), 
                useFixedBase=True)
        param = {
            "bodyUniqueId": ground,
            "linkIndex": -1,
        }
        if friction_coeff:
           param["lateralFriction"] = friction_coeff 
        if contact_stiffness and contact_damping:
           param["contactStiffness"] = contact_stiffness
           param["contactDamping"] = contact_damping
        self._pb_client.changeDynamics(**param)
        return ground

    def setup_others(self, friction_cone, solver_num_steps, solver_iter):
        ''' Dynamics parameters '''
        assert np.allclose(np.linalg.norm(self._v_up), 1.0)
        gravity = -9.8 * self._v_up
        self._pb_client.setGravity(gravity[0], gravity[1], gravity[2])
        self._pb_client.setTimeStep(self._dt_sim)
        self._pb_client.setPhysicsEngineParameter(numSubSteps=solver_num_steps)
        self._pb_client.setPhysicsEngineParameter(numSolverIterations=solver_iter)
        self._pb_client.setPhysicsEngineParameter(enableConeFriction=friction_cone)
        # self._pb_client.setPhysicsEngineParameter(solverResidualThreshold=1e-10)

    def check_collision(self, body_id1, body_id2, link_id1=None, link_id2=None):
        ''' collision between two bodies '''
        if link_id1 is None and link_id2 is None:
            pts = self._pb_client.getContactPoints(
                bodyA=body_id1, bodyB=body_id2)
        elif link_id1 is None and link_id2 is not None:
            pts = self._pb_client.getContactPoints(
                bodyA=body_id1, bodyB=body_id2, linkIndexB=link_id2)
        elif link_id1 is not None and link_id2 is None:
            pts = self._pb_client.getContactPoints(
                bodyA=body_id1, bodyB=body_id2, linkIndexA=link_id1)
        else:
            pts = self._pb_client.getContactPoints(
                bodyA=body_id1, bodyB=body_id2, linkIndexA=link_id1, linkIndexB=link_id2)
        return len(pts) > 0

    def check_falldown(self, idx):
        method = self._ground_falldown_check["method"]
        if method=="collision":
            return self.check_falldown_by_collision(idx)
        elif method=="height":
            height_min = self._ground_falldown_check["height_min"]
            return self.check_falldown_by_height(idx, height_min)
        else:
            raise NotImplementedError

    def check_falldown_by_collision(self, idx):
        agent = self._agent[idx]
        ''' check if any non-allowed body part hits the ground '''
        pts = self._pb_client.getContactPoints(bodyA=agent._body_id, bodyB=self._ground)
        for p in pts:
            part = p[3] if p[1] == agent._body_id else p[4]
            if agent._char_info.contact_allow_map[part]:
                continue
            else:
                return True
        return False

    def check_falldown_by_height(self, idx, height_min):
        agent = self._agent[idx]
        h = agent.get_root_height_from_ground(self.get_ground_height(idx))
        return h < height_min

    def get_ground_height_at(self, ps):
        assert isinstance(ps, list)
        dist = []
        if self._ground_height_map is not None:
            for p in ps:
                if self._v_up_str == "y":
                    idx1 = min(
                        int(self._ground_height_map_size[0]*(p[0]+0.5*self._ground_size[0])/self._ground_size[0]),
                        self._ground_height_map_size[0]-1)
                    idx2 = min(
                        int(self._ground_height_map_size[1]*(p[2]+0.5*self._ground_size[1])/self._ground_size[1]),
                        self._ground_height_map_size[1]-1)
                elif self._v_up_str == "z":
                    idx1 = min(
                        int(self._ground_height_map_size[0]*(p[0]+0.5*self._ground_size[0])/self._ground_size[0]),
                        self._ground_height_map_size[0]-1)
                    idx2 = min(
                        int(self._ground_height_map_size[1]*(p[1]+0.5*self._ground_size[1])/self._ground_size[1]),
                        self._ground_height_map_size[1]-1)
                else:
                    raise NotImplementedError
                dist.append(self._ground_height_map[idx1][idx2])
        else:
            ray_start = []
            ray_end = []
            for p in ps:
                '''
                If there exist object between ray_start and the ground,
                the height could be wrong.
                '''
                assert self._ground is not None
                if self._v_up_str == "y":
                    rs = np.array([p[0], -self._ground_height_measure_offset, p[2]])
                    re = np.array([p[0], self._ground_height_measure_offset, p[2]])
                elif self._v_up_str == "z":
                    rs = np.array([p[0], p[1], -self._ground_height_measure_offset])
                    re = np.array([p[0], p[1], self._ground_height_measure_offset])
                else:
                    raise NotImplementedError
                ray_start.append(rs)
                ray_end.append(re)

            res = self._pb_client.rayTestBatch(ray_start, ray_end)
            for i in range(len(res)):
                if res[i][0]==-1:
                    d = -self._ground_height_measure_offset
                else:
                    d = res[i][2]*2*self._ground_height_measure_offset
                    d -= self._ground_height_measure_offset
                dist.append(d)
        return dist

    def get_ground_height(self, idx):
        if self._use_default_ground:
            return 0.0
        return self.get_ground_height_at(
            [self._agent[idx].get_root_position()])[0]

    def get_ground_height_all(self):
        heights = []
        for i, agent in enumerate(self._sim_agent):
            heights.append(self.get_ground_height(i))
        return heights

    def check_sim_divergence(self, agent):
        ''' TODO: check divergence of simulation if necessary '''
        return False

    def check_out_of_ground(self, agent):
        p = agent.get_root_position()
        if self._v_up_str == "y":
            pos = np.array([p[0], p[2], p[1]])
        elif self._v_up_str == "z":
            pos = np.array([p[0], p[1], p[2]])
        else:
            raise NotImplementedError
        bb = 0.5*np.array(self._ground_size)
        aa = -bb
        return (pos > bb).any() or (pos < aa).any()

    def step(self, action_dict_list=None):
        ''' 
        One Step-forward Simulation 
        '''

        ''' Increase elapsed time '''
        self._elapsed_time += self._dt_con
        self._episode_len += self._dt_con

        if profile:
            print('++++++++++++++++++++++++++++++++++')
            tc.begin()

        if profile:
            act = []
            sim = []

        ''' Update simulation '''
        for _ in range(self._num_substep):
            if action_dict_list:
                for i in range(self._num_agent):
                    self._agent[i].actuate(action_dict_list[i])
            if profile:
                act.append(tc.get_time())
            self._pb_client.stepSimulation()
            if profile:
                sim.append(tc.get_time())

        if profile:
            print('actuation:', np.sum(act))
            print('simulation:', np.sum(sim))

        self._obs_manager.update()

        if profile:
            print('obs_manager.update:', tc.get_time())

        if profile:
            print('++++++++++++++++++++++++++++++++++')

    def reset(self, time=0.0, poses=None, vels=None, pb_state_id=None):

        ''' remove obstacles in the scene '''
        self._obs_manager.clear()

        ''' 
        Restore internal pybullet state 
        by uisng the saved info when Env was initially created  
        '''
        if pb_state_id is not None:
            self._pb_client.restoreState(pb_state_id)

        self._elapsed_time = time

        if poses is None:
            if self._init_state is not None:
                self._pb_client.restoreState(self._init_state)
        else:
            for i in range(self._num_agent):
                pose = poses[i]
                vel = None if vels is None else vels[i]
                self._agent[i].set_pose(pose, vel)

        self._episode_len = 0.0

    def add_noise_to_pose_vel(self, agent, pose, vel=None, return_as_copied=True):
        '''
        Add a little bit of noise to the given pose and velocity
        '''

        ref_pose = copy.deepcopy(pose) if return_as_copied else pose
        if vel:
            ref_vel = copy.deepcopy(vel) if return_as_copied else vel
        dof_cnt = 0
        for j in agent._joint_indices:
            joint_type = agent.get_joint_type(j)
            ''' Ignore fixed joints '''
            if joint_type == self._pb_client.JOINT_FIXED:
                continue
            ''' Ignore if there is no corresponding joint '''
            if agent._char_info.bvh_map[j] == None:
                continue
            T = ref_pose.get_transform(agent._char_info.bvh_map[j], local=True)
            R, p = conversions.T2Rp(T)
            if joint_type == self._pb_client.JOINT_SPHERICAL:
                dR = math.random_rotation(
                    mu_theta=agent._char_info.noise_pose[j][0],
                    sigma_theta=agent._char_info.noise_pose[j][1],
                    lower_theta=agent._char_info.noise_pose[j][2],
                    upper_theta=agent._char_info.noise_pose[j][3])
                dof_cnt += 3
            elif joint_type == self._pb_client.JOINT_REVOLUTE:
                theta = math.truncnorm(
                    mu=agent._char_info.noise_pose[j][0],
                    sigma=agent._char_info.noise_pose[j][1],
                    lower=agent._char_info.noise_pose[j][2],
                    upper=agent._char_info.noise_pose[j][3])
                joint_axis = agent.get_joint_axis(j)
                dR = conversions.A2R(joint_axis*theta)
                dof_cnt += 1
            else:
                raise NotImplementedError
            T_new = conversions.Rp2T(np.dot(R, dR), p)
            ref_pose.set_transform(agent._char_info.bvh_map[j], T_new, do_ortho_norm=False, local=True)
            if vel is not None:
                dw = math.truncnorm(
                    mu=np.full(3, agent._char_info.noise_vel[j][0]),
                    sigma=np.full(3, agent._char_info.noise_vel[j][1]),
                    lower=np.full(3, agent._char_info.noise_vel[j][2]),
                    upper=np.full(3, agent._char_info.noise_vel[j][3]))
                ref_vel.data_local[j][:3] += dw
        return ref_pose, ref_vel

    def get_render_data(self, agent):
        pb_client = self._pb_client
        model = agent._body_id
        joint_data = []
        link_data = []
        for j in range(pb_client.getNumJoints(model)):
            joint_info = pb_client.getJointInfo(model, j)
            joint_local_p, joint_local_Q, link_idx = joint_info[14], joint_info[15], joint_info[16]
            T_joint_local = conversions.Qp2T(
                np.array(joint_local_Q), np.array(joint_local_p))
            if link_idx == -1:
                link_world_p, link_world_Q = pb_client.getBasePositionAndOrientation(model)
            else:
                link_info = pb_client.getLinkState(model, link_idx)
                link_world_p, link_world_Q = link_info[0], link_info[1]
            T_link_world = conversions.Qp2T(
                np.array(link_world_Q), np.array(link_world_p))
            T_joint_world = np.dot(T_link_world, T_joint_local)
            R, p = conversions.T2Rp(T_joint_world)
            joint_data.append((conversions.R2Q(R), p))

        data_visual = pb_client.getVisualShapeData(model)
        lids = [d[1] for d in data_visual]
        dvs = data_visual
        for lid, dv in zip(lids, dvs):        
            if lid == -1:
                p, Q = pb_client.getBasePositionAndOrientation(model)
            else:
                link_state = pb_client.getLinkState(model, lid)
                p, Q = link_state[4], link_state[5]

            p, Q = np.array(p), np.array(Q)
            R = conversions.Q2R(Q)
            T_joint = conversions.Rp2T(R, p)
            T_visual_from_joint = \
                conversions.Qp2T(np.array(dv[6]),np.array(dv[5]))
            R, p = conversions.T2Rp(np.dot(T_joint, T_visual_from_joint))
            link_data.append((conversions.R2Q(R), p))

        return joint_data, link_data

    def render(self, rm):
        if self._num_agent == 1:
            colors = [rm.COLOR_AGENT]
        else:
            colors = rm.COLORS_FOR_AGENTS

        rm.gl.glEnable(rm.gl.GL_LIGHTING)
        rm.gl.glEnable(rm.gl.GL_BLEND)
        rm.gl.glBlendFunc(rm.gl.GL_SRC_ALPHA, rm.gl.GL_ONE_MINUS_SRC_ALPHA)

        for i in range(self._num_agent):
            sim_agent = self._agent[i]
            char_info = self._char_info[i]
            ground_height = self.get_ground_height(i)
            if rm.flag['sim_model']:
                rm.gl.glEnable(rm.gl.GL_DEPTH_TEST)
                if self._use_default_ground:
                    # This shadow render is only valid for the default ground
                    if rm.flag['shadow']:
                        rm.gl.glPushMatrix()
                        d = np.array([1, 1, 1])
                        d = d - math.projectionOnVector(d, self._v_up)
                        offset = (0.002 + ground_height) * self._v_up
                        rm.gl.glTranslatef(offset[0], offset[1], offset[2])
                        rm.gl.glScalef(d[0], d[1], d[2])
                        rm.bullet_render.render_model(
                            self._pb_client, 
                            sim_agent._body_id, 
                            draw_link=True, 
                            draw_link_info=False, 
                            draw_joint=False, 
                            draw_joint_geom=False, 
                            ee_indices=None, 
                            color=[0.5,0.5,0.5,1.0],
                            lighting=False)
                        rm.gl.glPopMatrix()
                rm.bullet_render.render_model(
                    self._pb_client, 
                    sim_agent._body_id,
                    draw_link=True, 
                    draw_link_info=True, 
                    draw_joint=rm.flag['joint'], 
                    draw_joint_geom=True, 
                    ee_indices=char_info.end_effector_indices, 
                    link_info_scale=rm.LINK_INFO_SCALE,
                    link_info_line_width=rm.LINK_INFO_LINE_WIDTH,
                    link_info_num_slice=rm.LINK_INFO_NUM_SLICE,
                    color=colors[i])
                if rm.flag['collision'] and self._elapsed_time > 0.0:
                    rm.gl.glPushAttrib(rm.gl.GL_LIGHTING|rm.gl.GL_DEPTH_TEST|rm.gl.GL_BLEND)
                    rm.gl.glEnable(rm.gl.GL_BLEND)
                    rm.bullet_render.render_contacts(self._pb_client, sim_agent._body_id)
                    rm.gl.glPopAttrib()
                if rm.flag['com_vel']:
                    p, Q, v, w = sim_agent.get_root_state()
                    p, v = sim_agent.get_com_and_com_vel()
                    rm.gl_render.render_arrow(p, p+v, D=0.01, color=[0, 0, 0, 1])
                if rm.flag['facing_frame']:
                    rm.gl.glPushAttrib(rm.gl.GL_LIGHTING|rm.gl.GL_DEPTH_TEST|rm.gl.GL_BLEND)
                    rm.gl.glEnable(rm.gl.GL_BLEND)
                    rm.gl_render.render_transform(
                        sim_agent.get_facing_transform(ground_height),
                        scale=0.5,
                        use_arrow=True)
                    rm.gl.glPopAttrib()

        if rm.flag['obstacle']:
            self._obs_manager.render()

if __name__ == '__main__':

    import env_renderer as er
    import render_module as rm
    from fairmotion.viz.utils import TimeChecker

    rm.initialize()

    class EnvRenderer(er.EnvRenderer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.time_checker_auto_play = TimeChecker()
            self.reset()
        def reset(self):
            self.env.reset()
        def one_step(self):
            self.env.step()
        def use_default_ground(self):
            return True
        def get_v_up_env_str(self):
            return self.env._v_up_str
        def get_ground(self):
            return self.env._ground
        def get_pb_client(self):
            return self.env._pb_client
        def extra_render_callback(self):
            self.env.render(self.rm)
        def extra_idle_callback(self):
            time_elapsed = self.time_checker_auto_play.get_time(restart=False)
            if self.rm.flag['auto_play'] and time_elapsed >= self.env._dt_con:
                self.time_checker_auto_play.begin()
                self.one_step()
        def extra_keyboard_callback(self, key):
            if key == b'r':
                self.reset()
            elif key == b'O':
                size = np.random.uniform(0.1, 0.3, 3)
                p, Q, v, w = self.env._agent[0].get_root_state()
                self.env._obs_manager.throw(p, size=size)
    
    print('=====Motion Tracking Controller=====')

    config = {
        'name': ['agent1'],
        'project_dir': './',
        'character': {
            'char_info_module': ['data/character/info/pfnn_char_info.py'],
            'sim_char_file': ['data/character/urdf/pfnn.urdf'],
            'ref_motion_scale': [1.0],
            'physics': [{'self_collision': True, 'actuation': 'spd'}],
        },
        'ground': {
            'enable': True,
            'falldown_check' : {
                'method': 'height',
                'height_min': 0.3,
            }
        }
    }

    env = Env(config)

    def default_cam(env):
        agent = env._agent[0]
        v_up_env = agent._char_info.v_up_env
        v_up = agent._char_info.v_up
        v_face = agent._char_info.v_face
        origin = np.zeros(3)
        return rm.camera.Camera(
            pos=3*(v_up+v_face),
            origin=origin, 
            vup=v_up_env, 
            fov=60.0)

    cam = default_cam(env)

    renderer = EnvRenderer(env=env, cam=cam)
    renderer.run()
