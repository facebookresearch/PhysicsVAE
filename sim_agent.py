# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from enum import Enum

from bullet import bullet_utils as bu

from fairmotion.ops import conversions
from fairmotion.ops import math
from fairmotion.ops import quaternion
from fairmotion.utils import constants
from fairmotion.core import motion

import warnings

profile = False
if profile:
    from fairmotion.viz.utils import TimeChecker
    tc = TimeChecker()

class SimAgent(object):
    '''
    This defines a simulated character in the scene.
    '''
    class Actuation(Enum):
        NONE=0  # No control
        SPD=1   # Stable PD Control
        PD=2    # PD Control
        CPD=3   # PD Control as Constraints of Simulation
        CP=4    # Position Control as Constraints of Simulation
        V=5     # Velocity Control as Constraints of Simulation
        TQ=6    # Torque Control
        @classmethod
        def from_string(cls, string):
            if string=="none": return cls.NONE
            if string=="spd": return cls.SPD
            if string=="pd": return cls.PD
            if string=="cpd": return cls.CPD
            if string=="cp": return cls.CP
            if string=="v": return cls.V
            if string=='tq': return cls.TQ
            raise NotImplementedError
    
    def __init__(self,
                 pybullet_client, 
                 model_file, 
                 char_info, 
                 scale=1.0, # This affects loadURDF 
                 ref_scale=1.0, # This will be used when reference motions are appllied to this agent
                 ref_height_fix=0.0,
                 verbose=False, 
                 kinematic_only=False,
                 self_collision=True,
                 name="agent",
                 actuation="none",
                 ):
        self._name = name
        self._actuation = SimAgent.Actuation.from_string(actuation)
        self._pb_client = pybullet_client
        self._char_info = char_info
        self._self_collision = self_collision
        # Load self._body_id file
        char_create_flags = self._pb_client.URDF_MAINTAIN_LINK_ORDER
        if self_collision:
            # char_create_flags = char_create_flags|\
            #                     self._pb_client.URDF_USE_SELF_COLLISION|\
            #                     self._pb_client.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
            char_create_flags = char_create_flags|\
                                self._pb_client.URDF_USE_SELF_COLLISION
        self._body_id = self._pb_client.loadURDF(
            model_file, 
            [0, 0, 0],
            globalScaling=scale,
            useFixedBase=False,
            flags=char_create_flags)
        for pair in self._char_info.collison_ignore_pairs:
            self._pb_client.setCollisionFilterPair(
                self._body_id,
                self._body_id,
                pair[0],
                pair[1],
                enableCollision=False)
        # TODO: should ref_scale be removed?
        self._ref_scale = ref_scale
        self._ref_height_fix = ref_height_fix
        self._ref_height_fix_v = \
            ref_height_fix * self._char_info.v_up_env
        self._num_joint = self._pb_client.getNumJoints(self._body_id)
        self._joint_indices = range(self._num_joint)
        self._link_indices = range(-1, self._num_joint)
        self._num_link = len(self._link_indices)
        self._joint_indices_movable = []
        if kinematic_only:
            self.setup_kinematics()
        else:
            self.setup_dynamics()
        # Pre-compute informations about the agent
        self._joint_type = []
        self._joint_axis = []
        self._joint_dofs = []
        for j in self._joint_indices:
            joint_info = self._pb_client.getJointInfo(self._body_id, j)
            self._joint_type.append(joint_info[2])
            self._joint_axis.append(np.array(joint_info[13]))
            # print('-----------------------')
            # print(joint_info[1])
            # print('joint_type', joint_info[2])
            # print('joint_damping', joint_info[6])
            # print('joint_friction', joint_info[7])
            # print('joint_upper_limit', joint_info[8])
            # print('joint_lower_limit', joint_info[9])
            # print('joint_max_force', joint_info[10])
            # print('joint_max_vel', joint_info[11])
        for j in self._joint_indices:
            if self._joint_type[j] == self._pb_client.JOINT_SPHERICAL:
                self._joint_dofs.append(3)
                self._joint_indices_movable.append(j)
            elif self._joint_type[j] == self._pb_client.JOINT_REVOLUTE: 
                self._joint_dofs.append(1)
                self._joint_indices_movable.append(j)
            elif self._joint_type[j] == self._pb_client.JOINT_FIXED: 
                self._joint_dofs.append(0)
            else:
                raise NotImplementedError()
        self._num_dofs = np.sum(self._joint_dofs)
        self._joint_pose_init, self._joint_vel_init = self.get_joint_states()
        self._joint_parent_link = []
        self._joint_xform_from_parent_link = []
        for j in self._joint_indices:
            joint_info = self._pb_client.getJointInfo(self._body_id, j)
            joint_local_p = np.array(joint_info[14])
            joint_local_Q = np.array(joint_info[15])
            link_idx = joint_info[16]
            self._joint_parent_link.append(link_idx)
            self._joint_xform_from_parent_link.append(
                conversions.Qp2T(joint_local_Q, joint_local_p))
        self._link_masses = []
        self._link_total_mass = 0.0
        for i in self._link_indices:
            di = self._pb_client.getDynamicsInfo(self._body_id, i)
            mass = di[0]
            self._link_total_mass += mass
            self._link_masses.append(mass)
        # print('total_mass', self._link_total_mass)
        self._act_joint_indices = []
        self._act_target_positions = []
        self._act_target_velocities = []
        self._act_kps = []
        self._act_kds = []
        self._act_max_forces = []
        self._act_all_spherical_joints = True
        for j in self._joint_indices:
            joint_type = self.get_joint_type(j)
            if joint_type == self._pb_client.JOINT_FIXED:
                ''' Ignore fixed joints '''
                continue
            self._act_joint_indices.append(j)
            ''' Collect target poses and vels '''
            self._act_target_positions.append(self._joint_pose_init[j])
            self._act_target_velocities.append(self._joint_vel_init[j])
            ''' Collect max forces '''
            if joint_type == self._pb_client.JOINT_REVOLUTE:
                max_force = np.array([self._char_info.max_force[j]])
                self._act_all_spherical_joints = False
            elif joint_type == self._pb_client.JOINT_SPHERICAL:
                max_force = np.ones(3) * self._char_info.max_force[j]
            else:
                raise NotImplementedError
            ''' Collect control gains '''
            if self._actuation==SimAgent.Actuation.SPD:
                self._act_kps.append(self._char_info.kp['spd'][j])
                self._act_kds.append(self._char_info.kd['spd'][j])
            elif self._actuation==SimAgent.Actuation.PD:
                self._act_kps.append(self._char_info.kp['pd'][j])
                self._act_kds.append(self._char_info.kd['pd'][j])
            elif self._actuation==SimAgent.Actuation.CPD or \
                 self._actuation==SimAgent.Actuation.CP or \
                 self._actuation==SimAgent.Actuation.V:
                self._act_kps.append(self._char_info.kp['cpd'][j])
                self._act_kds.append(self._char_info.kd['cpd'][j])
            self._act_max_forces.append(max_force)

        # print(self._act_joint_indices)
        # print(self._act_target_positions)
        # print(self._act_target_velocities)
        # print(self._act_kps)
        # print(self._act_kds)
        # print(self._act_max_forces)
        # exit(0)
    
    def get_name(self):
        return self._name
    
    def split_joint_variables(self, states, joint_indices):
        states_out = []
        idx = 0
        for j in joint_indices:
            joint_type = self._joint_type[j]
            if joint_type == self._pb_client.JOINT_SPHERICAL:
                Q = conversions.A2Q(np.array(states[idx:idx+3]))
                states_out.append(Q)
                idx += 3
            elif joint_type == self._pb_client.JOINT_REVOLUTE: 
                states_out.append([states[idx]])
                idx += 1
            elif joint_type == self._pb_client.JOINT_FIXED: 
                pass
            else:
                raise NotImplementedError()
        assert idx == len(states)
        return states_out
    
    def setup_dynamics(self):
        # Settings for the simulation self._body_id
        for j in self._link_indices:
            self._pb_client.changeDynamics(
                self._body_id, 
                j, 
                lateralFriction=self._char_info.friction_lateral, 
                spinningFriction=self._char_info.friction_spinning,
                restitution=self._char_info.restitution)
            di = self._pb_client.getDynamicsInfo(self._body_id, j)
        # Disable the initial motor control
        for j in self._joint_indices:
            self._pb_client.setJointMotorControl2(
                self._body_id, 
                j, 
                self._pb_client.POSITION_CONTROL, 
                targetPosition=0,
                targetVelocity=0,
                force=0)
            self._pb_client.setJointMotorControlMultiDof(
                self._body_id, 
                j, 
                self._pb_client.POSITION_CONTROL,
                targetPosition=[0,0,0,1], 
                targetVelocity=[0,0,0], 
                positionGain=0, 
                velocityGain=0, 
                force=[0,0,0])
        for j in self._joint_indices:
            self._pb_client.enableJointForceTorqueSensor(self._body_id, j, enableSensor=True)
    
    def setup_kinematics(self):
        # Settings for the kinematic self._body_id so that it does not affect the simulation
        self._pb_client.changeDynamics(self._body_id, -1, linearDamping=0, angularDamping=0)
        self._pb_client.setCollisionFilterGroupMask(
            self._body_id,
            -1,
            collisionFilterGroup=0,
            collisionFilterMask=0)
        for j in range(-1, self._pb_client.getNumJoints(self._body_id)):
            self._pb_client.setCollisionFilterGroupMask(
                self._body_id,
                j,
                collisionFilterGroup=0,
                collisionFilterMask=0)
            self._pb_client.changeDynamics(
                self._body_id,
                j,
                activationState=self._pb_client.ACTIVATION_STATE_SLEEP +
                self._pb_client.ACTIVATION_STATE_ENABLE_SLEEPING +
                self._pb_client.ACTIVATION_STATE_DISABLE_WAKEUP)
        self.change_visual_color([0, 1, 0, 0])
    
    def change_visual_color(self, color):
        self._pb_client.changeVisualShape(self._body_id, -1, rgbaColor=color)
        for j in range(self._pb_client.getNumJoints(self._body_id)):
            self._pb_client.changeVisualShape(self._body_id, j, rgbaColor=color)
    
    def get_num_dofs(self):
        return self._num_dofs
    
    def get_num_joint(self):
        return self._num_joint
    
    def get_joint_type(self, idx):
        return self._joint_type[idx]
    
    def get_joint_axis(self, idx):
        return self._joint_axis[idx]
    
    def get_joint_dofs(self, idx):
        return self._joint_dofs[idx]
    
    def get_root_height_from_ground(self, ground_height):
        p, _, _, _ = bu.get_base_pQvw(self._pb_client, self._body_id)
        vec_root_from_ground = math.projectionOnVector(p, self._char_info.v_up_env)
        return np.linalg.norm(vec_root_from_ground) - ground_height
    
    def get_root_state(self):
        return bu.get_base_pQvw(self._pb_client, self._body_id)
    
    def get_root_transform(self):
        p, Q, _, _ = bu.get_base_pQvw(self._pb_client, self._body_id)
        return conversions.Qp2T(Q, p)

    def get_root_position(self):
        p, Q, _, _ = bu.get_base_pQvw(self._pb_client, self._body_id)
        return p
    
    def set_root_transform(self, T):
        Q, p = conversions.T2Qp(T)
        bu.set_base_pQvw(self._pb_client, self._body_id, p, Q, None, None)
    
    def get_facing_transform(self, ground_height):
        d, p = self.get_facing_direction_position(ground_height)
        z = d
        y = self._char_info.v_up_env
        x = np.cross(y, z)
        return conversions.Rp2T(np.array([x, y, z]).transpose(), p)       
    
    def get_facing_position(self, ground_height):
        d, p = self.get_facing_direction_position(ground_height)
        return p
    
    def get_facing_direction(self):
        d, p = self.get_facing_direction_position(0)
        return d
    
    def get_facing_direction_position(self, ground_height):
        R, p = conversions.T2Rp(self.get_root_transform())
        d = np.dot(R, self._char_info.v_face)
        if np.allclose(d, self._char_info.v_up_env):
            msg = \
                '\n+++++++++++++++++WARNING+++++++++++++++++++\n'+\
                'The facing direction is ill-defined ' +\
                '(i.e. parellel to the world up-vector).\n'+\
                'A random direction will be assigned for the direction\n'+\
                'Be careful if your system is sensitive to the facing direction\n'+\
                '+++++++++++++++++++++++++++++++++++++++++++\n'
            warnings.warn(msg)
            d = math.random_unit_vector()
        d = d - math.projectionOnVector(d, self._char_info.v_up_env)
        p = p - math.projectionOnVector(p, self._char_info.v_up_env)
        if ground_height != 0.0:
            p += ground_height * self._char_info.v_up_env
        return d/np.linalg.norm(d), p
    
    def project_to_ground(self, v):
        return v - math.projectionOnVector(v, self._char_info.v_up_env)
    
    def get_link_states(self, indices=None):
        return bu.get_link_pQvw(self._pb_client, self._body_id, indices)
    
    def get_joint_states(self, indices=None):
        return bu.get_joint_pv(self._pb_client, self._body_id, indices)
    
    def set_pose_by_xform(self, xform):
        assert len(xform) == len(self._char_info.bvh_map_inv)

        ''' Base '''
        Q, p = conversions.T2Qp(xform[0])
        p *= self._ref_scale
        p += self._ref_height_fix_v

        bu.set_base_pQvw(self._pb_client, self._body_id, p, Q, None, None)

        ''' Others '''
        indices = []
        state_pos = []
        state_vel = []
        idx = -1
        for k, j in self._char_info.bvh_map_inv.items():
            idx += 1
            if idx == 0: continue
            if j is None: continue
            joint_type = self._joint_type[j]
            if joint_type == self._pb_client.JOINT_FIXED:
                continue
            T = xform[idx]
            if joint_type == self._pb_client.JOINT_SPHERICAL:
                Q, p = conversions.T2Qp(T)
                w = np.zeros(3)
                state_pos.append(Q)
                state_vel.append(w)
            elif joint_type == self._pb_client.JOINT_REVOLUTE:
                joint_axis = self.get_joint_axis(j)
                R, p = conversions.T2Rp(T)
                w = np.zeros(3)
                state_pos.append(math.project_rotation_1D(R, joint_axis))
                state_vel.append(math.project_angular_vel_1D(w, joint_axis))
            else:
                raise NotImplementedError()
            indices.append(j)
        
        bu.set_joint_pv(self._pb_client, self._body_id, indices, state_pos, state_vel)

    def set_pose(self, pose, vel=None):
        '''
        Velocity should be represented w.r.t. local frame
        '''
        # Root joint
        T = pose.get_transform(
            self._char_info.bvh_map[self._char_info.ROOT], 
            local=False)
        Q, p = conversions.T2Qp(T)
        p *= self._ref_scale
        p += self._ref_height_fix_v
        
        v, w = None, None
        if vel is not None:
            # Here we give a root orientation to get velocities represeted in world frame.
            R = conversions.Q2R(Q)
            w = vel.get_angular(
                self._char_info.bvh_map[self._char_info.ROOT], False, R)
            v = vel.get_linear(
                self._char_info.bvh_map[self._char_info.ROOT], False, R)
            v *= self._ref_scale

        bu.set_base_pQvw(self._pb_client, self._body_id, p, Q, v, w)

        # Other joints
        indices = []
        state_pos = []
        state_vel = []
        for j in self._joint_indices:
            joint_type = self._joint_type[j]
            # When the target joint do not have dof, we simply ignore it
            if joint_type == self._pb_client.JOINT_FIXED:
                continue
            # When there is no matching between the given pose and the simulated character,
            # the character just tries to hold its initial pose
            if self._char_info.bvh_map[j] is None:
                state_pos.append(self._joint_pose_init[j])
                state_vel.append(self._joint_vel_init[j])
            else:
                T = pose.get_transform(self._char_info.bvh_map[j], local=True)            
                if joint_type == self._pb_client.JOINT_SPHERICAL:
                    Q, p = conversions.T2Qp(T)
                    w = np.zeros(3) if vel is None else vel.get_angular(self._char_info.bvh_map[j], local=True)
                    state_pos.append(Q)
                    state_vel.append(w)
                elif joint_type == self._pb_client.JOINT_REVOLUTE:
                    joint_axis = self.get_joint_axis(j)
                    R, p = conversions.T2Rp(T)
                    w = np.zeros(3) if vel is None else vel.get_angular(self._char_info.bvh_map[j], local=True)
                    state_pos.append([math.project_rotation_1D(R, joint_axis)])
                    state_vel.append([math.project_angular_vel_1D(w, joint_axis)])
                else:
                    raise NotImplementedError()
            indices.append(j)
        bu.set_joint_pv(self._pb_client, self._body_id, indices, state_pos, state_vel)

    def get_pose_data(self, skel, apply_height_offset=True):
        p, Q, v, w, ps, vs = bu.get_state_all(self._pb_client, self._body_id)
        pose_data = []
        for i in range(skel.num_joints()):
            joint = skel.joints[i]
            if joint == skel.root_joint:
                if not apply_height_offset:
                    p -= self._ref_height_fix_v
                pose_data.append(conversions.Qp2T(Q, p))
            else:
                j = self._char_info.bvh_map_inv[joint.name]
                if j is None:
                    pose_data.append(constants.eye_T())
                else:
                    joint_type = self._joint_type[j]
                    if joint_type == self._pb_client.JOINT_FIXED:
                        pose_data.append(constants.eye_T())
                    elif joint_type == self._pb_client.JOINT_SPHERICAL:
                        pose_data.append(conversions.Q2T(ps[j]))
                    else:
                        raise NotImplementedError()
        return pose_data
    
    def get_pose(self, skel, apply_height_offset=True):
        return motion.Pose(skel, self.get_pose_data(skel, apply_height_offset))
    
    def array_to_pose_data(self, skel, data, T_root_ref=None):
        assert len(data) == self._num_dofs + 6
        T_root = conversions.Rp2T(conversions.A2R(data[3:6]), data[0:3])
        if T_root_ref is not None:
            T_root = np.dot(T_root_ref, T_root)
        pose_data = []
        idx = 6
        for i in range(skel.num_joints()):
            joint = skel.joints[i]
            if joint == skel.root_joint:
                pose_data.append(T_root)
            else:
                j = self._char_info.bvh_map_inv[joint.name]
                if j is None:
                    pose_data.append(constants.eye_T())
                else:
                    joint_type = self._joint_type[j]
                    if joint_type == self._pb_client.JOINT_FIXED:
                        pose_data.append(constants.eye_T())
                    elif joint_type == self._pb_client.JOINT_SPHERICAL:
                        pose_data.append(conversions.R2T(conversions.A2R(data[idx:idx+3])))
                        idx += 3
                    else:
                        raise NotImplementedError()
        return pose_data
    
    def arrary_to_pose(self, skel, data, T_root_ref=None):
        pose_data = self.array_to_pose_data(skel, data)
        return motion.Pose(skel, pose_data)
    
    def save_states(self):
        return bu.get_state_all(self._pb_client, self._body_id)
    
    def restore_states(self, states):
        bu.set_state_all(self._pb_client, self._body_id, states)
    
    def get_com_and_com_vel(self):
        return bu.compute_com_and_com_vel(self._pb_client, self._body_id, self._link_indices)
    
    def get_joint_torques(self):
        return bu.get_joint_torques(self._pb_client, self._body_id, self._joint_indices)
    
    def get_joint_weights(self, skel):
        ''' Get joint weight values form char_info '''
        joint_weights = []
        for j in skel.joints:
            idx = self._char_info.bvh_map_inv[j.name]
            if idx is None:
                joint_weights.append(0.0)
            else:
                w = self._char_info.joint_weight[idx]
                joint_weights.append(w)
        return np.array(joint_weights)
    
    def interaction_mesh_samples(self):
        assert self._char_info.interaction_mesh_samples is not None
        def get_joint_position(j, p_root, Q_root, p_link, Q_link):
            if j==self._char_info.ROOT or self._joint_parent_link[j]==self._char_info.ROOT:
                p, Q = p_root, Q_root
            else:
                p, Q = p_link[self._joint_parent_link[j]], Q_link[self._joint_parent_link[j]]
            T_link_world = conversions.Qp2T(Q, p)
            T_joint_local = constants.eye_T() if j==self._char_info.ROOT else self._joint_xform_from_parent_link[j]
            T_joint_world = np.dot(T_link_world, T_joint_local)
            return conversions.T2p(T_joint_world)
        points = []
        p_root, Q_root, _, _ = self.get_root_state()
        p_link, Q_link, _, _ = self.get_link_states()
        for j1, j2, alpha in self._char_info.interaction_mesh_samples:
            p1 = get_joint_position(j1, p_root, Q_root, p_link, Q_link)
            p2 = p1 if j2 is None else get_joint_position(j2, p_root, Q_root, p_link, Q_link)
            points.append((1.0 - alpha) * p1 + alpha * p2)
        return points
    
    def inverse_kinematics(self, indices, positions):
        assert len(indices) == len(positions)
        new_positions = self._pb_client.calculateInverseKinematics2(
            self._body_id, 
            endEffectorLinkIndices=indices, 
            targetPositions=positions,
            solver=0,
            maxNumIterations=100,
            residualThreshold=.01)
        # new_positions = self._pb_client.calculateInverseKinematics(self._body_id, self._char_info.RightHand, np.zeros(3))
        new_positions = self.split_joint_variables(new_positions, self._joint_indices_movable)
        for p in new_positions:
            print(p)
        self._pb_client.resetJointStatesMultiDof(self._body_id, 
                                                 self._joint_indices_movable, 
                                                 new_positions)

    def actuate(self, action_dict):
        pose = action_dict.get("pose")
        vel = action_dict.get("vel")
        torque = action_dict.get("torque")
        base_residual_linear_force = action_dict.get("base_residual_linear_force")
        base_residual_angular_force = action_dict.get("base_residual_angular_force")
        base_residual_linear_force_frame = action_dict.get("base_residual_linear_force_frame")
        base_residual_angular_force_frame = action_dict.get("base_residual_angular_force_frame")

        if profile:
            print('**********************************')
            tc.begin()

        if self._actuation == SimAgent.Actuation.NONE:
            return

        if base_residual_linear_force is not None:
            if base_residual_linear_force_frame=="base":
                R = conversions.T2R(self.get_root_transform())
            elif base_residual_linear_force_frame=="facing":
                R = conversions.T2R(self.get_facing_transform(0))
            elif base_residual_linear_force_frame=="world":
                R = constants.EYE_R
            else:
                raise NotImplementedError
            f = np.dot(R, base_residual_linear_force)
            self._pb_client.applyExternalForce(
                self._body_id,
                -1,
                f,
                self.get_root_position(),
                self._pb_client.WORLD_FRAME)

        if base_residual_angular_force is not None:
            if base_residual_angular_force_frame=="base":
                R = constants.EYE_R
                # R = conversions.T2R(self.get_root_transform())
            # elif base_residual_angular_force_frame=="facing":
            #     R = conversions.T2R(self.get_facing_transform(0))
            # elif base_residual_angular_force_frame=="world":
            #     R = constants.EYE_R
            else:
                raise NotImplementedError
            f = np.dot(R, base_residual_angular_force)
            self._pb_client.applyExternalTorque(
                self._body_id,
                -1,
                f,
                self._pb_client.LINK_FRAME)

        if self._actuation != SimAgent.Actuation.TQ:
            if self._act_all_spherical_joints:
                '''
                If all actuable joints are spherical (we do not have to check joint types), 
                so we can take a shortcut
                '''
                Ts, ws = [], []
                for idx, j in enumerate(self._act_joint_indices):
                    T = pose.get_transform(self._char_info.bvh_map[j], local=True)
                    if vel is None:
                        w = constants.ZERO_P
                    else: 
                        w = vel.get_angular(self._char_info.bvh_map[j], local=True)
                    Ts.append(T)
                    ws.append(w)
                Qs, ps = conversions.T2Qp(np.array(Ts))
                self._act_target_positions = Qs
                self._act_target_velocities = ws
                # for idx, j in enumerate(self._act_joint_indices):
                #     self._act_target_positions[idx] = Qs[idx]
                #     self._act_target_velocities[idx] = ws[idx]
            else:
                '''
                Otherwise, we need to check the type of the joint and
                do appropriate operations according to the type
                '''
                for idx, j in enumerate(self._act_joint_indices):
                    joint_type = self.get_joint_type(j)
                    if self._char_info.bvh_map[j] == None:
                        ''' Use the initial pose if no mapping exists '''
                        target_pos = self._joint_pose_init[j]
                        target_vel = self._joint_vel_init[j]
                    else:
                        ''' 
                        Convert target pose value so that it fits to each joint type
                        For the hinge joint, we find the geodesic closest value given the axis
                        '''
                        if pose is None:
                            T = constants.EYE_T
                        else:
                            T = pose.get_transform(self._char_info.bvh_map[j], local=True)              
                        if vel is None:
                            w = constants.ZERO_P
                        else:
                            w = vel.get_angular(self._char_info.bvh_map[j], local=True)
                        if joint_type == self._pb_client.JOINT_REVOLUTE:
                            axis = self.get_joint_axis(j)
                            target_pos = np.array([math.project_rotation_1D(conversions.T2R(T), axis)])
                            target_vel = np.array([math.project_angular_vel_1D(w, axis)])
                        elif joint_type == self._pb_client.JOINT_SPHERICAL:
                            Q, p = conversions.T2Qp(T)
                            # Q = quaternion.Q_op(Q, op=["normalize", "halfspace"])
                            target_pos = Q
                            target_vel = w
                        else:
                            raise NotImplementedError
                    self._act_target_positions[idx] = target_pos
                    self._act_target_velocities[idx] = target_vel
        else:
            if self._act_all_spherical_joints:
                forces = torque.reshape((-1, 3))
            else:
                forces = []
                cnt = 0
                for idx, j in enumerate(self._act_joint_indices):
                    joint_type = self.get_joint_type(j)
                    if joint_type == self._pb_client.JOINT_REVOLUTE:
                        forces.append(torque[cnt:cnt+1])
                        cnt += 1
                    elif joint_type == self._pb_client.JOINT_SPHERICAL:
                        forces.append(torque[cnt:cnt+3])
                        cnt += 3
                    else:
                        raise NotImplementedError

        if profile:
            print('actuation_setup:', tc.get_time())

        if self._actuation==SimAgent.Actuation.SPD:
            self._pb_client.setJointMotorControlMultiDofArray(
                self._body_id,
                self._act_joint_indices,
                self._pb_client.STABLE_PD_CONTROL,
                targetPositions=self._act_target_positions,
                targetVelocities=self._act_target_velocities,
                forces=self._act_max_forces,
                positionGains=self._act_kps,
                velocityGains=self._act_kds)
        elif self._actuation==SimAgent.Actuation.PD:
            ''' Basic PD in Bullet does not support spherical joint yet '''
            # self._pb_client.setJointMotorControlMultiDofArray(
            #     self._body_id,
            #     self._act_joint_indices,
            #     self._pb_client.PD_CONTROL,
            #     targetPositions=self._act_target_positions,
            #     targetVelocities=self._act_target_velocities,
            #     forces=self._act_max_forces,
            #     positionGains=self._act_kps,
            #     velocityGains=self._act_kds)
            # self._pb_client.setJointMotorControlMultiDofArray(
            #     self._body_id,
            #     joint_indices,
            #     self._pb_client.PD_CONTROL,
            #     targetPositions=self._act_target_positions,
            #     targetVelocities=self._act_target_velocities,
            #     forces=self._act_max_forces,
            #     positionGains=kps,
            #     velocityGains=kds)
            forces = bu.compute_PD_forces(
                pb_client=self._pb_client,
                body_id=self._body_id, 
                joint_indices=self._act_joint_indices, 
                desired_positions=self._act_target_positions, 
                desired_velocities=self._act_target_velocities, 
                kps=self._act_kps, 
                kds=self._act_kds,
                max_forces=self._act_max_forces)
            self._pb_client.setJointMotorControlMultiDofArray(
                self._body_id,
                self._act_joint_indices,
                self._pb_client.TORQUE_CONTROL,
                forces=forces)
        elif self._actuation==SimAgent.Actuation.CPD:
            self._pb_client.setJointMotorControlMultiDofArray(
                self._body_id,
                self._act_joint_indices,
                self._pb_client.POSITION_CONTROL,
                targetPositions=self._act_target_positions,
                targetVelocities=self._act_target_velocities,
                forces=self._act_max_forces,
                positionGains=self._act_kps,
                velocityGains=self._act_kds)
        elif self._actuation==SimAgent.Actuation.CP:
            self._pb_client.setJointMotorControlMultiDofArray(
                self._body_id,
                self._act_joint_indices,
                self._pb_client.POSITION_CONTROL,
                targetPositions=self._act_target_positions,
                forces=self._act_max_forces,
                positionGains=self._act_kps)
        elif self._actuation==SimAgent.Actuation.V:
            self._pb_client.setJointMotorControlMultiDofArray(
                self._body_id,
                self._act_joint_indices,
                self._pb_client.VELOCITY_CONTROL,
                targetVelocities=self._act_target_velocities,
                forces=self._act_max_forces,
                velocityGains=self._act_kds)
        elif self._actuation==SimAgent.Actuation.TQ:
            self._pb_client.setJointMotorControlMultiDofArray(
                self._body_id,
                self._act_joint_indices,
                self._pb_client.TORQUE_CONTROL,
                forces=forces)
        else:
            raise NotImplementedError

        # print(bu.get_joint_torques(self._pb_client, self._body_id, [10]))

        if profile:
            print('actuation_run_bullet:', tc.get_time())

        if profile:
            print('**********************************')
    
    def actuate2(self, pose=None, vel=None, torque=None):

        if profile:
            print('**********************************')
            tc.begin()

        if self._actuation == SimAgent.Actuation.NONE:
            return
        joint_indices = []
        target_positions = []
        target_velocities = []
        kps = []
        kds = []
        max_forces = []
        if profile:
            part1 = []
            part2 = []
            part3 = []
        for j in self._joint_indices:
            joint_type = self.get_joint_type(j)
            if joint_type == self._pb_client.JOINT_FIXED:
                ''' Ignore fixed joints '''
                continue
            joint_indices.append(j)
            if self._actuation==SimAgent.Actuation.TQ:
                ''' No need to compute target values for torque control '''
                continue
            if profile:
                part1.append(tc.get_time())
            ''' Collect target poses and vels '''
            if self._char_info.bvh_map[j] == None:
                ''' Use the initial pose if no mapping exists '''
                target_pos = self._joint_pose_init[j]
                target_vel = self._joint_vel_init[j]
            else:
                ''' 
                Convert target pose value so that it fits to each joint type
                For the hinge joint, we find the geodesic closest value given the axis
                '''
                if pose is None:
                    T = constants.eye_T()
                else:
                    T = pose.get_transform(self._char_info.bvh_map[j], local=True)              
                if vel is None:
                    w = np.zeros(3)
                else:
                    w = vel.get_angular(self._char_info.bvh_map[j])
                if joint_type == self._pb_client.JOINT_REVOLUTE:
                    axis = self.get_joint_axis(j)
                    target_pos = np.array([math.project_rotation_1D(conversions.T2R(T), axis)])
                    target_vel = np.array([math.project_angular_vel_1D(w, axis)])
                elif joint_type == self._pb_client.JOINT_SPHERICAL:
                    Q, p = conversions.T2Qp(T)
                    Q = quaternion.Q_op(Q, op=["normalize", "halfspace"])
                    target_pos = Q
                    target_vel = w
                else:
                    raise NotImplementedError
            target_positions.append(target_pos)
            target_velocities.append(target_vel)
            if profile:
                part2.append(tc.get_time())
            ''' Collect max forces '''
            if joint_type == self._pb_client.JOINT_REVOLUTE:
                max_force = np.array([self._char_info.max_force[j]])
            elif joint_type == self._pb_client.JOINT_SPHERICAL:
                max_force = np.ones(3) * self._char_info.max_force[j]
            else:
                raise NotImplementedError
            ''' Collect control gains '''
            if self._actuation==SimAgent.Actuation.SPD:
                kps.append(self._char_info.kp[j])
                kds.append(self._char_info.kd[j])
            elif self._actuation==SimAgent.Actuation.PD:
                kps.append(self._char_info.kp[j])
                kds.append(self._char_info.kd[j])
            elif self._actuation==SimAgent.Actuation.CPD or \
                 self._actuation==SimAgent.Actuation.CP or \
                 self._actuation==SimAgent.Actuation.V:
                kps.append(self._char_info.cpd_ratio*self._char_info.kp[j])
                kds.append(self._char_info.cpd_ratio*self._char_info.kd[j])
            max_forces.append(max_force)
            if profile:
                part3.append(tc.get_time())

        if profile:
            p1 = np.sum(part1)
            p2 = np.sum(part2)
            p3 = np.sum(part3)
            print('part1:', p1)
            print('part2:', p2)
            print('part3:', p3)
            print('actuation_setup:', p1+p2+p3)

        if self._actuation==SimAgent.Actuation.SPD:
            self._pb_client.setJointMotorControlMultiDofArray(self._body_id,
                                                              joint_indices,
                                                              self._pb_client.STABLE_PD_CONTROL,
                                                              targetPositions=target_positions,
                                                              targetVelocities=target_velocities,
                                                              forces=max_forces,
                                                              positionGains=kps,
                                                              velocityGains=kds)
        elif self._actuation==SimAgent.Actuation.PD:
            ''' Basic PD in Bullet does not support spherical joint yet '''
            self._pb_client.setJointMotorControlMultiDofArray(self._body_id,
                                                              joint_indices,
                                                              self._pb_client.PD_CONTROL,
                                                              targetPositions=target_positions,
                                                              targetVelocities=target_velocities,
                                                              forces=max_forces,
                                                              positionGains=kps,
                                                              velocityGains=kds)
            # forces = bu.compute_PD_forces(pb_client=self._pb_client,
            #                               body_id=self._body_id, 
            #                               joint_indices=joint_indices, 
            #                               desired_positions=target_positions, 
            #                               desired_velocities=target_velocities, 
            #                               kps=kps, 
            #                               kds=kds,
            #                               max_forces=max_forces)
            # self._pb_client.setJointMotorControlMultiDofArray(self._body_id,
            #                                                   joint_indices,
            #                                                   self._pb_client.TORQUE_CONTROL,
            #                                                   forces=forces)
        elif self._actuation==SimAgent.Actuation.CPD:
            self._pb_client.setJointMotorControlMultiDofArray(self._body_id,
                                                              joint_indices,
                                                              self._pb_client.POSITION_CONTROL,
                                                              targetPositions=target_positions,
                                                              targetVelocities=target_velocities,
                                                              forces=max_forces,
                                                              positionGains=kps,
                                                              velocityGains=kds)
        elif self._actuation==SimAgent.Actuation.CP:
            self._pb_client.setJointMotorControlMultiDofArray(self._body_id,
                                                              joint_indices,
                                                              self._pb_client.POSITION_CONTROL,
                                                              targetPositions=target_positions,
                                                              forces=max_forces,
                                                              positionGains=kps)
        elif self._actuation==SimAgent.Actuation.V:
            self._pb_client.setJointMotorControlMultiDofArray(self._body_id,
                                                              joint_indices,
                                                              self._pb_client.VELOCITY_CONTROL,
                                                              targetVelocities=target_velocities,
                                                              forces=max_forces,
                                                              velocityGains=kds)
        elif self._actuation==SimAgent.Actuation.TQ:
            self._pb_client.setJointMotorControlMultiDofArray(self._body_id,
                                                              joint_indices,
                                                              self._pb_client.TORQUE_CONTROL,
                                                              forces=torque)
        else:
            raise NotImplementedError

        if profile:
            print('actuation_run_bullet:', tc.get_time())

        if profile:
            print('**********************************')

