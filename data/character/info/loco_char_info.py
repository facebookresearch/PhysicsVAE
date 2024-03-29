# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import collections
import numpy as np

name = "LOCOMOTION"

''' 
The up direction of the character w.r.t. its root joint.
The up direction in the world frame can be computed by dot(R_root, v_up), 
where R_root is the orientation of the root.
'''
v_up = np.array([0.0, 1.0, 0.0])
''' 
The facing direction of the character w.r.t. its root joint.
The facing direction in the world frame can be computed by dot(R_root, v_face), 
where R_root is the orientation of the root.
'''
v_face = np.array([0.0, 0.0, 1.0])
''' 
The up direction of the world frame, when the character holds its defalult posture (e.g. t-pose).
This information is useful/necessary when comparing a relationship between the character and its environment.
'''
v_up_env = np.array([0.0, 1.0, 0.0])
v_ax1_env = np.array([0.0, 0.0, 1.0])
v_ax2_env = np.array([1.0, 0.0, 0.0])

''' 
Definition of Link/Joint (In our character definition, one joint can only have one link)
'''
Hips = -1
LHipJoint = 0
LeftUpLeg = 1
LeftLeg = 2
LeftFoot = 3
LeftToeBase = 4
RHipJoint = 5
RightUpLeg = 6
RightLeg = 7
RightFoot = 8
RightToeBase = 9
LowerBack = 10
Spine = 11
Spine1 = 12
Neck = 13
Neck1 = 14
Head = 15
LeftShoulder = 16
LeftArm = 17
LeftForeArm = 18
LeftHand = 19
RightShoulder = 20
RightArm = 21
RightForeArm = 22
RightHand = 23

''' 
Definition of the root (base) joint
'''
ROOT = Hips

''' 
Definition of end effectors
'''
end_effector_indices = [
    LeftHand, 
    RightHand, 
    LeftFoot, 
    RightFoot
]

''' 
Mapping from joint indicies to names
'''
joint_name = collections.OrderedDict()

joint_name[Hips] = "Hips"
joint_name[LHipJoint] = "LHipJoint"
joint_name[LeftUpLeg] = "LeftUpLeg"
joint_name[LeftLeg] = "LeftLeg"
joint_name[LeftFoot] = "LeftFoot"
joint_name[LeftToeBase] = "LeftToeBase"
joint_name[RHipJoint] = "RHipJoint"
joint_name[RightUpLeg] = "RightUpLeg"
joint_name[RightLeg] = "RightLeg"
joint_name[RightFoot] = "RightFoot"
joint_name[RightToeBase] = "RightToeBase"
joint_name[LowerBack] = "LowerBack"
joint_name[Spine] = "Spine"
joint_name[Spine1] = "Spine1"
joint_name[Neck] = "Neck"
joint_name[Neck1] = "Neck1"
joint_name[Head] = "Head"
joint_name[LeftShoulder] = "LeftShoulder"
joint_name[LeftArm] = "LeftArm"
joint_name[LeftForeArm] = "LeftForeArm"
joint_name[LeftHand] = "LeftHand"
joint_name[RightShoulder] = "RightShoulder"
joint_name[RightArm] = "RightArm"
joint_name[RightForeArm] = "RightForeArm"
joint_name[RightHand] = "RightHand"

''' 
Mapping from joint names to indicies
'''
joint_idx = collections.OrderedDict()

joint_idx["Hips"] = Hips
joint_idx["LHipJoint"] = LHipJoint
joint_idx["LeftUpLeg"] = LeftUpLeg
joint_idx["LeftLeg"] = LeftLeg
joint_idx["LeftFoot"] = LeftFoot
joint_idx["LeftToeBase"] = LeftToeBase
joint_idx["RHipJoint"] = RHipJoint
joint_idx["RightUpLeg"] = RightUpLeg
joint_idx["RightLeg"] = RightLeg
joint_idx["RightFoot"] = RightFoot
joint_idx["RightToeBase"] = RightToeBase
joint_idx["LowerBack"] = LowerBack
joint_idx["Spine"] = Spine
joint_idx["Spine1"] = Spine1
joint_idx["Neck"] = Neck
joint_idx["Neck1"] = Neck1
joint_idx["Head"] = Head
joint_idx["LeftShoulder"] = LeftShoulder
joint_idx["LeftArm"] = LeftArm
joint_idx["LeftForeArm"] = LeftForeArm
joint_idx["LeftHand"] = LeftHand
joint_idx["RightShoulder"] = RightShoulder
joint_idx["RightArm"] = RightArm
joint_idx["RightForeArm"] = RightForeArm
joint_idx["RightHand"] = RightHand


''' 
Mapping from character's joint indicies to bvh's joint names.
Some entry could have no mapping (by assigning None).
'''
bvh_map = collections.OrderedDict()

bvh_map[Hips]            = 'Hips'
bvh_map[LHipJoint]       = 'LHipJoint'
bvh_map[LeftUpLeg]       = 'LeftUpLeg'
bvh_map[LeftLeg]         = 'LeftLeg'
bvh_map[LeftFoot]        = 'LeftFoot'
bvh_map[LeftToeBase]     = None
bvh_map[RHipJoint]       = 'RHipJoint'
bvh_map[RightUpLeg]      = 'RightUpLeg'
bvh_map[RightLeg]        = 'RightLeg'
bvh_map[RightFoot]       = 'RightFoot'
bvh_map[RightToeBase]    = None
bvh_map[LowerBack]       = 'LowerBack'
bvh_map[Spine]           = 'Spine'
bvh_map[Spine1]          = 'Spine1'
bvh_map[Neck]            = 'Neck'
bvh_map[Neck1]           = 'Neck1'
bvh_map[Head]            = 'Head'
bvh_map[LeftShoulder]    = 'LeftShoulder'
bvh_map[LeftArm]         = 'LeftArm'
bvh_map[LeftForeArm]     = 'LeftForeArm'
bvh_map[LeftHand]        = None
bvh_map[RightShoulder]   = 'RightShoulder'
bvh_map[RightArm]        = 'RightArm'
bvh_map[RightForeArm]    = 'RightForeArm'
bvh_map[RightHand]       = None

''' 
Mapping from bvh's joint names to character's joint indicies.
Some entry could have no mapping (by assigning None).
'''
bvh_map_inv = collections.OrderedDict()

bvh_map_inv['Hips']            = Hips
bvh_map_inv['LHipJoint']       = LHipJoint
bvh_map_inv['LeftUpLeg']       = LeftUpLeg
bvh_map_inv['LeftLeg']         = LeftLeg
bvh_map_inv['LeftFoot']        = LeftFoot
bvh_map_inv['LeftToeBase']     = None
bvh_map_inv['RHipJoint']       = RHipJoint
bvh_map_inv['RightUpLeg']      = RightUpLeg
bvh_map_inv['RightLeg']        = RightLeg
bvh_map_inv['RightFoot']       = RightFoot
bvh_map_inv['RightToeBase']    = None
bvh_map_inv['LowerBack']       = LowerBack
bvh_map_inv['Spine']           = Spine
bvh_map_inv['Spine1']          = Spine1
bvh_map_inv['Neck']            = Neck
bvh_map_inv['Neck1']           = Neck1
bvh_map_inv['Head']            = Head
bvh_map_inv['LeftShoulder']    = LeftShoulder
bvh_map_inv['LeftArm']         = LeftArm
bvh_map_inv['LeftForeArm']     = LeftForeArm
bvh_map_inv['LeftHand']        = None
bvh_map_inv['LeftFingerBase']  = None
bvh_map_inv['LeftHandIndex1']  = None
bvh_map_inv['LThumb']          = None
bvh_map_inv['RightShoulder']   = RightShoulder
bvh_map_inv['RightArm']        = RightArm
bvh_map_inv['RightForeArm']    = RightForeArm
bvh_map_inv['RightHand']       = None
bvh_map_inv['RightFingerBase'] = None
bvh_map_inv['RightHandIndex1'] = None
bvh_map_inv['RThumb']          = None

dof = {
    Hips : 6,
    LHipJoint : 4,
    LeftUpLeg : 4,
    LeftLeg : 4,
    LeftFoot : 4,
    LeftToeBase : 0,
    RHipJoint : 4,
    RightUpLeg : 4,
    RightLeg : 4,
    RightFoot : 4,
    RightToeBase : 0,
    LowerBack : 4,
    Spine : 4,
    Spine1 : 4,
    Neck : 0,
    Neck1 : 4,
    Head : 0,
    LeftShoulder : 4,
    LeftArm : 4,
    LeftForeArm : 4,
    LeftHand : 0,
    RightShoulder : 4,
    RightArm : 4,
    RightForeArm : 4,
    RightHand : 0,
    }


''' 
Definition of PD gains
'''

kp = {}
kd = {}

kp['spd'] = {
    Hips : 0,
    LHipJoint : 500,
    LeftUpLeg : 500,
    LeftLeg : 500,
    LeftFoot : 500,
    LeftToeBase : 0,
    RHipJoint : 500,
    RightUpLeg : 500,
    RightLeg : 500,
    RightFoot : 500,
    RightToeBase : 0,
    LowerBack : 500,
    Spine : 500,
    Spine1 : 500,
    Neck : 500,
    Neck1 : 500,
    Head : 500,
    LeftShoulder : 500,
    LeftArm : 500,
    LeftForeArm : 500,
    LeftHand : 0,
    RightShoulder : 500,
    RightArm : 500,
    RightForeArm : 500,
    RightHand : 0,
    }

kd['spd'] = {}
for k, v in kp['spd'].items():
    kd['spd'][k] = 0.1 * v
kd['spd'][ROOT] = 0

''' 
Definition of PD gains (tuned for Contrained PD Controller).
"cpd_ratio * kp" and "cpd_ratio * kd" will be used respectively.
'''
cpd_ratio = 0.0002


''' 
Maximum forces that character can generate when PD controller is used.
'''
max_force = {
    Hips : 0,
    LHipJoint : 1000,
    LeftUpLeg : 1000,
    LeftLeg : 800,
    LeftFoot : 800,
    LeftToeBase : 0,
    RHipJoint : 1000,
    RightUpLeg : 1000,
    RightLeg : 800,
    RightFoot : 800,
    RightToeBase : 0,
    LowerBack : 800,
    Spine : 800,
    Spine1 : 800,
    Neck : 500,
    Neck1 : 500,
    Head : 500,
    LeftShoulder : 500,
    LeftArm : 500,
    LeftForeArm : 500,
    LeftHand : 0,
    RightShoulder : 500,
    RightArm : 500,
    RightForeArm : 500,
    RightHand : 0,
    }

contact_allow_map = {
    Hips : False,
    LHipJoint : False,
    LeftUpLeg : False,
    LeftLeg : True,
    LeftFoot : True,
    LeftToeBase : True,
    RHipJoint : False,
    RightUpLeg : False,
    RightLeg : True,
    RightFoot : True,
    RightToeBase : True,
    LowerBack : False,
    Spine : False,
    Spine1 : False,
    Neck : False,
    Neck1 : False,
    Head : False,
    LeftShoulder : False,
    LeftArm : False,
    LeftForeArm : False,
    LeftHand : False,
    RightShoulder : False,
    RightArm : False,
    RightForeArm : False,
    RightHand : False,
    }

joint_weight = {
    Hips : 1.0,
    LHipJoint : 0.5,
    LeftUpLeg : 0.5,
    LeftLeg : 0.3,
    LeftFoot : 0.2,
    LeftToeBase : 0.1,
    RHipJoint : 0.5,
    RightUpLeg : 0.5,
    RightLeg : 0.3,
    RightFoot : 0.2,
    RightToeBase : 0.1,
    LowerBack : 0.5,
    Spine : 0.5,
    Spine1 : 0.3,
    Neck : 0.3,
    Neck1 : 0.3,
    Head : 0.3,
    LeftShoulder : 0.5,
    LeftArm : 0.5,
    LeftForeArm : 0.3,
    LeftHand : 0.2,
    RightShoulder : 0.5,
    RightArm : 0.5,
    RightForeArm : 0.3,
    RightHand : 0.2,
    }

sum_joint_weight = 0.0
for key, val in joint_weight.items():
    sum_joint_weight += val
for key, val in joint_weight.items():
    joint_weight[key] /= sum_joint_weight

collison_ignore_pairs = [
    (Spine, LeftShoulder),
    (Spine, RightShoulder),
    (Spine1, LeftShoulder),
    (Spine1, RightShoulder),
    (Neck, LeftShoulder),
    (Neck, RightShoulder),
    (LowerBack, LeftShoulder),
    (LowerBack, RightShoulder),
    (LHipJoint, RHipJoint),
    (LHipJoint, LowerBack),
    (RHipJoint, LowerBack),
    (LHipJoint, Spine),
    (RHipJoint, Spine),
    (LeftShoulder, RightShoulder),
    (Neck, Head),
]

friction_lateral = 0.8
friction_spinning = 0.0
restitution = 0.0