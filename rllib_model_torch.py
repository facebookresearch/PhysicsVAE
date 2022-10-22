# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import numpy as np
import os

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.utils.annotations import override
# from ray.rllib.utils.framework import get_activation_fn
from ray.rllib.utils import try_import_torch
from ray.rllib.models import ModelCatalog

torch, nn = try_import_torch()

import torch.nn.functional as F

logger = logging.getLogger(__name__)

def softmax_normalized(x, dim):
    x_hat = x-torch.max(x, dim=dim)[0].unsqueeze(-1)
    return F.softmax(x_hat, dim=dim)

def get_activation_fn(name=None):
    
    if name in ["linear", None]:
        return None
    if name in ["swish", "silu"]:
        from ray.rllib.utils.torch_ops import Swish
        return Swish
    if name == "relu":
        return nn.ReLU
    if name == "tanh":
        return nn.Tanh
    if name == "sigmoid":
        return nn.Sigmoid
    if name == "elu":
        return nn.ELU

    raise ValueError("Unknown activation ({})={}!".format(name))

def create_layer(layer_type, layers, size_in, size_out, append_log_std=False):
    output_layer = None
    lstm_layer = None
    if layer_type == "mlp":
        param = {
            "size_in": size_in, 
            "size_out": size_out, 
            "layers": layers,
            "append_log_std": append_log_std,
        }
        output_layer = FC(**param)
    elif layer_type == "lstm":
        ## TODO: needed to be fixed by layers
        assert layers[0]["type"] == "lstm"
        hidden_size = layers[0]["hidden_size"]
        num_layers = layers[0]["num_layers"]
        lstm_layer = nn.LSTM(
            input_size=size_in,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        ''' For the compatibility of the previous config '''
        output_activation = layers[0].get("output_activation")
        if output_activation:
            if output_activation == "linear":
                output_layer = nn.Sequential(
                    nn.Linear(
                        in_features=hidden_size,
                        out_features=size_out,
                    ),
                )
            elif output_activation == "tanh":
                output_layer = nn.Sequential(
                    nn.Linear(
                        in_features=hidden_size,
                        out_features=size_out,
                    ),
                    nn.Tanh(),
                )
            else:
                raise NotImplementedError

        output_layers = layers[0].get("output_layers")
        if output_layers:
            param = {
                "size_in": hidden_size, 
                "size_out": size_out, 
                "layers": output_layers,
                "append_log_std": append_log_std,
            }
            output_layer = FC(**param)
    else:
        raise NotImplementedError
    assert output_layer is not None
    return output_layer, lstm_layer

def forward_layer(obs, seq_lens=None, state=None, state_cnt=None, output_layer=None, lstm_layer=None):
    if lstm_layer is None and output_layer is None:
        out = obs
    elif lstm_layer is None and output_layer is not None:
        out = output_layer(obs)
    elif lstm_layer is not None and output_layer is not None:
        assert seq_lens is not None and state is not None and state_cnt is not None
        out, state_cnt = process_lstm(obs, seq_lens, state, state_cnt, output_layer, lstm_layer)
    else:
        raise Exception("Invalid Inputs")
    return out, state_cnt

def process_lstm(obs, seq_lens, state, state_cnt, output_layer, lstm_layer):
    if isinstance(seq_lens, np.ndarray):
        seq_lens = torch.Tensor(seq_lens).int()
    if seq_lens is not None:
        max_seq_len = obs.shape[0] // seq_lens.shape[0]
        # max_seq_len=torch.max(seq_lens)
    
    input_lstm = add_time_dimension(
        obs,
        max_seq_len=max_seq_len,
        framework="torch",
        time_major=False,
    )

    ''' 
    Assume that the shape of state is 
    (batch, num_layers * num_directions, hidden_size). So we change
    the first axis with the second axis.
    '''
    h_lstm, c_lstm = state[state_cnt], state[state_cnt+1]

    h_lstm = h_lstm.reshape(h_lstm.shape[1], h_lstm.shape[0], h_lstm.shape[2])
    c_lstm = c_lstm.reshape(c_lstm.shape[1], c_lstm.shape[0], c_lstm.shape[2])
    
    output_lstm, (h_lstm, c_lstm) = lstm_layer(input_lstm, (h_lstm, c_lstm))
    output_lstm = output_lstm.reshape(-1, output_lstm.shape[-1])
    out = output_layer(output_lstm)

    '''
    Change the first and second axes of the output state so that
    it matches to the assumption
    '''
    h_lstm = h_lstm.reshape(h_lstm.shape[1], h_lstm.shape[0], h_lstm.shape[2])
    c_lstm = c_lstm.reshape(c_lstm.shape[1], c_lstm.shape[0], c_lstm.shape[2])

    state[state_cnt] = h_lstm
    state[state_cnt+1] = c_lstm

    state_cnt += 2

    return out, state_cnt

class AppendLogStd(nn.Module):
    '''
    An appending layer for log_std.
    '''
    def __init__(self, type, init_val, dim):
        super().__init__()
        self.type = type

        if np.isscalar(init_val):
            init_val = init_val * np.ones(dim)
        elif isinstance(init_val, (np.ndarray, list)):
            assert len(init_val) == dim
        else:
            raise NotImplementedError

        self.init_val = init_val

        if self.type=="constant":
            self.log_std = torch.Tensor(init_val)
        elif self.type=="state_independent":
            self.log_std = torch.nn.Parameter(
                torch.Tensor(init_val))
            self.register_parameter("log_std", self.log_std)
        else:
            raise NotImplementedError

    def set_val(self, val):
        assert self.type=="constant", \
            "Change value is only allowed in constant logstd"
        assert np.isscalar(val), \
            "Only scalar is currently supported"

        self.log_std[:] = val
    
    def forward(self, x):
        assert x.shape[-1] == self.log_std.shape[-1]
        
        shape = list(x.shape)
        for i in range(0, len(shape)-1):
            shape[i] = 1
        log_std = torch.reshape(self.log_std, shape)
        shape = list(x.shape)
        shape[-1] = 1
        log_std = log_std.repeat(shape)

        out = torch.cat([x, log_std], axis=-1)
        return out

class Hardmax(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
    def forward(self, x):
        idx = torch.argmax(x, dim=1)
        # print(x.shape)
        # print(idx, self.num_classes)
        y = F.one_hot(idx, num_classes=self.num_classes)
        # print(y)
        return y

def get_initializer(info):
    if info['name'] == "normc":
        return normc_initializer(info['std'])
    elif info['name'] == 'xavier_normal':
        def initializer(tensor):
            return nn.init.xavier_normal_(tensor, gain=info['gain'])
        return initializer
    elif info['name'] == 'xavier_uniform':
        def initializer(tensor):
            return nn.init.xavier_uniform_(tensor, gain=info['gain'])
        return initializer
    else:
        raise NotImplementedError

class FC(nn.Module):
    ''' 
    A network with fully connected layers.
    '''
    def __init__(self, size_in, size_out, layers, append_log_std=False,
                 log_std_type='constant', sample_std=1.0):
        super().__init__()
        nn_layers = []
        prev_layer_size = size_in
        for l in layers:
            layer_type = l['type']
            if layer_type == 'fc':
                assert isinstance(l['hidden_size'] , int) or l['hidden_size'] =='output'
                hidden_size = l['hidden_size'] if l['hidden_size'] != 'output' else size_out
                layer = SlimFC(
                    in_size=prev_layer_size,
                    out_size=hidden_size,
                    initializer=get_initializer(l['init_weight']),
                    activation_fn=get_activation_fn(l['activation'])
                )
                prev_layer_size = hidden_size
            elif layer_type in ['bn', 'batch_norm']:
                layer = nn.BatchNorm1d(prev_layer_size)
            elif layer_type in ['sm', 'softmax']:
                layer = nn.Softmax(dim=1)
            elif layer_type in ['hm', 'hardmax']:
                layer = Hardmax(num_classes=prev_layer_size)
            else:
                raise NotImplementedError(
                    "Unknown Layer Type:", layer_type)
            nn_layers.append(layer)

        if append_log_std:
            nn_layers.append(AppendLogStd(
                type=log_std_type, 
                init_val=np.log(sample_std), 
                dim=size_out))

        self._model = nn.Sequential(*nn_layers)
    
    def forward(self, x):
        return self._model(x)

    def save_weights(self, file):
        torch.save(self.state_dict(), file)

    def load_weights(self, file):
        self.load_state_dict(torch.load(file))
        self.eval()


DEFAULT_FC_64X2 = [
    {"type": "fc", "hidden_size": 64, "activation": "relu", "init_weight": {"name": "normc", "std": 1.0}},
    {"type": "fc", "hidden_size": 64, "activation": "relu", "init_weight": {"name": "normc", "std": 1.0}},
    {"type": "fc", "hidden_size": "output", "activation": "linear", "init_weight": {"name": "normc", "std": 0.01}},
]

DEFAULT_FC_128X2 = [
    {"type": "fc", "hidden_size": 128, "activation": "relu", "init_weight": {"name": "normc", "std": 1.0}},
    {"type": "fc", "hidden_size": 128, "activation": "relu", "init_weight": {"name": "normc", "std": 1.0}},
    {"type": "fc", "hidden_size": "output", "activation": "linear", "init_weight": {"name": "normc", "std": 0.01}},
]

DEFAULT_FC_256X2 = [
    {"type": "fc", "hidden_size": 256, "activation": "relu", "init_weight": {"name": "normc", "std": 1.0}},
    {"type": "fc", "hidden_size": 256, "activation": "relu", "init_weight": {"name": "normc", "std": 1.0}},
    {"type": "fc", "hidden_size": "output", "activation": "linear", "init_weight": {"name": "normc", "std": 0.01}},
]

DEFAULT_FC_512X2 = [
    {"type": "fc", "hidden_size": 512, "activation": "relu", "init_weight": {"name": "normc", "std": 1.0}},
    {"type": "fc", "hidden_size": 512, "activation": "relu", "init_weight": {"name": "normc", "std": 1.0}},
    {"type": "fc", "hidden_size": "output", "activation": "linear", "init_weight": {"name": "normc", "std": 0.01}},
]

DEFAULT_FC_512X3 = [
    {"type": "fc", "hidden_size": 512, "activation": "relu", "init_weight": {"name": "normc", "std": 1.0}},
    {"type": "fc", "hidden_size": 512, "activation": "relu", "init_weight": {"name": "normc", "std": 1.0}},
    {"type": "fc", "hidden_size": 512, "activation": "relu", "init_weight": {"name": "normc", "std": 1.0}},
    {"type": "fc", "hidden_size": "output", "activation": "linear", "init_weight": {"name": "normc", "std": 0.01}},
]

DEFAULT_FC_1024X2 = [
    {"type": "fc", "hidden_size": 1024, "activation": "relu", "init_weight": {"name": "normc", "std": 1.0}},
    {"type": "fc", "hidden_size": 1024, "activation": "relu", "init_weight": {"name": "normc", "std": 1.0}},
    {"type": "fc", "hidden_size": "output", "activation": "linear", "init_weight": {"name": "normc", "std": 0.01}},
]


class FullyConnectedPolicy(TorchModelV2, nn.Module):
    ''' 
    A policy that generates action and value with FCNN
    '''
    DEFAULT_CONFIG = {
        "log_std_type": "constant",
        "sample_std": 1.0,

        "policy_fn_type": "mlp",
        "policy_fn_layers": DEFAULT_FC_256X2,

        "log_std_fn_layers": DEFAULT_FC_64X2,
        
        "value_fn_layers": DEFAULT_FC_256X2,
    }
    """Generic fully connected network."""
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name, **model_kwargs):

        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        ''' Load and check configuarations '''

        custom_model_config = FullyConnectedPolicy.DEFAULT_CONFIG.copy()
        custom_model_config_by_user = model_config.get("custom_model_config")
        if custom_model_config_by_user:
            custom_model_config.update(custom_model_config_by_user)

        '''
        constant
            log_std will not change during the training
        state_independent
            log_std will be learned during the training
            but it does not depend on the state of the agent
        state_dependent:
            log_std will be learned during the training
            and it depens on the state of the agent
        '''

        log_std_type = custom_model_config.get("log_std_type")
        assert log_std_type in \
            ["constant", "state_independent", "state_dependent"]

        sample_std = custom_model_config.get("sample_std")
        assert np.array(sample_std).all() > 0.0, \
            "The value shoulde be positive"

        assert num_outputs % 2 == 0, (
            "num_outputs must be divisible by two", num_outputs)
        num_outputs = num_outputs//2
        append_log_std = (log_std_type != "state_dependent")

        policy_fn_type = custom_model_config.get("policy_fn_type")
        policy_fn_layers = custom_model_config.get("policy_fn_layers")

        log_std_fn_layers = custom_model_config.get("log_std_fn_layers")
        
        value_fn_layers = custom_model_config.get("value_fn_layers")

        dim_state = int(np.product(obs_space.shape))

        ''' Construct the policy function '''

        if policy_fn_type == "mlp":
            param = {
                "size_in": dim_state, 
                "size_out": num_outputs, 
                "layers": policy_fn_layers,
                "append_log_std": append_log_std,
                "log_std_type": log_std_type,
                "sample_std": sample_std
            }
            self._policy_fn = FC(**param)
        else:
            raise NotImplementedError

        ''' Construct the value function '''

        param = {
            "size_in": dim_state, 
            "size_out": 1, 
            "layers": value_fn_layers,
            "append_log_std": False
        }
        self._value_fn = FC(**param)

        ''' Keep the latest output of the value function '''

        self._cur_value = None

        ''' Construct log_std function if necessary '''

        self._log_std_fn = None

        if log_std_type == "state_dependent":
            param = {
                "size_in": dim_state,
                "size_out": num_outputs,
                "layers": log_std_fn_layers,
                "append_log_std": False,
            }
            self._log_std_fn = FC(**param)
            self._log_std_base = np.log(sample_std)

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs_flat"].float()
        obs = obs.reshape(obs.shape[0], -1)
        
        logits = self._policy_fn(obs)
        self._cur_value = self._value_fn(obs).squeeze(1)

        if self._log_std_fn is not None:
            log_std = self._log_std_base + self._log_std_fn(obs)
            logits = torch.cat([logits, log_std], axis=-1)
            
        return logits, state

    @override(TorchModelV2)
    def value_function(self):
        assert self._cur_value is not None, "must call forward() first"
        return self._cur_value

    def set_exploration_std(self, std):
        log_std = np.log(std)
        self._policy_fn._model[-1].set_val(log_std)

    def save_policy_weights(self, file):
        torch.save(self._policy_fn.state_dict(), file)

    def load_policy_weights(self, file):
        self._policy_fn.load_state_dict(torch.load(file))
        self._policy_fn.eval()



class PhysicsVAE(TorchModelV2, nn.Module):
    DEFAULT_CONFIG = {
        "project_dir": None,
        
        "log_std_type": "constant",
        "sample_std": 1.0,

        "load_weights": None,

        "task_encoder_inputs": ["body", "task"],
        "task_encoder_layers": DEFAULT_FC_256X2,
        "task_encoder_load_weights": None,
        "task_encoder_learnable": True,
        "task_encoder_output_dim": 32,
        
        # The prior distribution for the latent space
        ## False: no prior
        ## normal_zero_mean_one_std: zero mean, one std
        ## normal_state_mean_one_std: state conditioned mean, one std
        ## hypersphere_uniform: zero center, one radius
        "latent_prior_type": "normal_zero_mean_one_std",
        # This is used when we use a learnable prior only
        "latent_prior_layers": None,

        "motor_decoder_inputs": ["body", "task"],
        "motor_decoder_layers": DEFAULT_FC_512X3,
        "motor_decoder_load_weights": None,
        "motor_decoder_learnable": True,

        "motor_decoder_helper_enable": False,
        "motor_decoder_helper_layers": [
            {"type": "fc", "hidden_size": 128, "activation": "relu", "init_weight": {"name": "normc", "std": 1.0}},
            {"type": "fc", "hidden_size": 128, "activation": "relu", "init_weight": {"name": "normc", "std": 1.0}},
            {"type": "fc", "hidden_size": "output", "activation": "tanh", "init_weight": {"name": "normc", "std": 0.01}},
        ],
        "motor_decoder_helper_load_weights": None,
        "motor_decoder_helper_learnable": True,
        "motor_decoder_helper_range": 0.5,
        
        "value_fn_layers": DEFAULT_FC_256X2,

        "world_model_layers": DEFAULT_FC_1024X2,
        "world_model_load_weights": None,
        "world_model_learnable": True,

        "observation_space": None,
        "observation_space_body": None,
        "observation_space_task": None,
        "action_space": None,
    }
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name, **model_kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        ''' Load and check configuarations '''

        assert num_outputs % 2 == 0, (
            "num_outputs must be divisible by two", num_outputs)
        num_outputs = num_outputs // 2

        custom_model_config = PhysicsVAE.DEFAULT_CONFIG.copy()
        custom_model_config_by_user = model_config.get("custom_model_config")
        if custom_model_config_by_user:
            custom_model_config.update(custom_model_config_by_user)

        log_std_type = custom_model_config.get("log_std_type")
        assert log_std_type in ["constant", "state_independent"]
        sample_std = custom_model_config.get("sample_std")
        
        load_weights = custom_model_config.get("load_weights")

        project_dir = custom_model_config.get("project_dir")

        task_encoder_inputs = custom_model_config.get("task_encoder_inputs")
        task_encoder_layers = custom_model_config.get("task_encoder_layers")
        task_encoder_output_dim = custom_model_config.get("task_encoder_output_dim")
        task_encoder_load_weights = custom_model_config.get("task_encoder_load_weights")
        task_encoder_learnable = custom_model_config.get("task_encoder_learnable")
        task_encoder_autoreg = custom_model_config.get("task_encoder_autoreg")
        task_encoder_autoreg_alpha = custom_model_config.get("task_encoder_autoreg_alpha")
        
        latent_prior_type = custom_model_config.get("latent_prior_type")
        latent_prior_layers = custom_model_config.get("latent_prior_layers")

        self._task_encoder_inputs = task_encoder_inputs
        self._task_encoder_layers = task_encoder_layers
        self._task_encoder_output_dim = task_encoder_output_dim
        self._task_encoder_autoreg = task_encoder_autoreg
        self._task_encoder_autoreg_alpha = task_encoder_autoreg_alpha
        
        self._latent_prior_type = latent_prior_type
        self._latent_prior_layers = latent_prior_layers

        motor_decoder_inputs = custom_model_config.get("motor_decoder_inputs")
        motor_decoder_layers = custom_model_config.get("motor_decoder_layers")
        motor_decoder_load_weights = custom_model_config.get("motor_decoder_load_weights")
        motor_decoder_learnable = custom_model_config.get("motor_decoder_learnable")

        motor_decoder_helper_enable = custom_model_config.get("motor_decoder_helper_enable")
        motor_decoder_helper_layers = custom_model_config.get("motor_decoder_helper_layers")
        motor_decoder_helper_load_weights = custom_model_config.get("motor_decoder_helper_load_weights")
        motor_decoder_helper_learnable = custom_model_config.get("motor_decoder_helper_learnable")
        motor_decoder_helper_range = custom_model_config.get("motor_decoder_helper_range")

        self._motor_decoder_inputs = motor_decoder_inputs
        self._motor_decoder_helper_range = motor_decoder_helper_range

        value_fn_layers = custom_model_config.get("value_fn_layers")
        self._value_fn_layers = value_fn_layers

        world_model_layers = custom_model_config.get("world_model_layers")
        world_model_load_weights = custom_model_config.get("world_model_load_weights")
        world_model_learnable = custom_model_config.get("world_model_learnable")

        if project_dir:
            if load_weights:
                load_weights = \
                    os.path.join(project_dir, load_weights)
                assert load_weights
            if task_encoder_load_weights:
                task_encoder_load_weights = \
                    os.path.join(project_dir, task_encoder_load_weights)
                assert task_encoder_load_weights
            if motor_decoder_load_weights:
                motor_decoder_load_weights = \
                    os.path.join(project_dir, motor_decoder_load_weights)
                assert motor_decoder_load_weights
            if motor_decoder_helper_load_weights:
                motor_decoder_helper_load_weights = \
                    os.path.join(project_dir, motor_decoder_helper_load_weights)
                assert motor_decoder_helper_load_weights
            if world_model_load_weights:
                world_model_load_weights = \
                    os.path.join(project_dir, world_model_load_weights)
                assert world_model_load_weights

        self.dim_state_body = \
            int(np.product(custom_model_config.get("observation_space_body").shape))
        self.dim_state_task = \
            int(np.product(custom_model_config.get("observation_space_task").shape))
        self.dim_state = int(np.product(obs_space.shape))
        self.dim_action = int(np.product(action_space.shape))

        assert self.dim_state == self.dim_state_body + self.dim_state_task

        size_in_task_encoder = 0
        if "body" in self._task_encoder_inputs:
            size_in_task_encoder += self.dim_state_body
        if "task" in self._task_encoder_inputs:
            size_in_task_encoder += self.dim_state_task

        # normal_zero_mean_one_std: zero mean, one std
        # normal_state_mean_one_std: state conditioned mean, one std
        # hypersphere_uniform: zero center, one radius

        if latent_prior_type in ["normal_zero_mean_one_std", "normal_state_mean_one_std"]:
            size_out_task_encoder = 2*task_encoder_output_dim
        elif latent_prior_type in ["hypersphere_uniform"]:
            size_out_task_encoder = task_encoder_output_dim
        elif latent_prior_type == False:
            size_out_task_encoder = task_encoder_output_dim
        else:
            raise NotImplementedError("Unknown latent_prior_type:"+latent_prior_type)

        if latent_prior_type in ["normal_state_mean_one_std"]:
            self._latent_prior, _ = \
                create_layer(
                    layer_type="mlp",
                    layers=latent_prior_type,
                    size_in=self.dim_state_body,
                    size_out=task_encoder_output_dim,
                    )

        ''' Prepare task encoder that outputs task embedding z given s_task '''
        self._task_encoder, _ = \
            create_layer(
                layer_type="mlp",
                layers=task_encoder_layers,
                size_in=size_in_task_encoder,
                size_out=size_out_task_encoder,
                )

        size_in_motor_decoder_body = 0
        size_in_motor_decoder_task = 0
        if "body" in self._motor_decoder_inputs:
            size_in_motor_decoder_body = self.dim_state_body
        if "task" in self._motor_decoder_inputs:
            size_in_motor_decoder_task = task_encoder_output_dim if self._task_encoder else self.dim_state_task
        size_in_motor_decoder = size_in_motor_decoder_body + size_in_motor_decoder_task
        assert size_in_motor_decoder > 0

        ''' Prepare motor control decoder that outputs a given (z, s_proprioception) '''

        def motor_decoder_fn():
            param = {
                "size_in": size_in_motor_decoder,
                "size_out": num_outputs, 
                "layers": motor_decoder_layers,
                "append_log_std": True,
                "log_std_type": log_std_type, 
                "sample_std": sample_std,
            } 
            return FC(**param)

        self._motor_decoder = motor_decoder_fn()

        self._motor_decoder_helper = None
        if motor_decoder_helper_enable:
            assert motor_decoder_helper_layers[-1]["activation"] == "tanh"
            assert motor_decoder_helper_range > 0
            param = {
                "size_in": size_in_motor_decoder,
                "size_out": num_outputs, 
                "layers": motor_decoder_helper_layers,
                "append_log_std": False,
            }
            self._motor_decoder_helper = FC(**param)

        self._world_model = None
        param = {
            "size_in": self.dim_action + self.dim_state_body,
            "size_out": self.dim_state_body, 
            "layers": world_model_layers,
            "append_log_std": False,
        }
        self._world_model = FC(**param)

        ''' Prepare a value function '''

        self._value_branch, _ = \
            create_layer(
                layer_type="mlp",
                layers=value_fn_layers,
                size_in=self.dim_state,
                size_out=1,
                )

        self._cur_value = None
        self._cur_task_encoder_variable = None
        self._cur_task_encoder_mu = None
        self._cur_task_encoder_logvar = None
        self.latent_prior_noise = True

        ''' Load pre-trained weight if exists '''

        if load_weights:
            self.load_weights(load_weights)
            print("load_weights:", load_weights)

        if task_encoder_load_weights:
            self.load_weights_task_encoder(task_encoder_load_weights)
            self.set_learnable_task_encoder(task_encoder_learnable)

        if motor_decoder_load_weights:
            self.load_weights_motor_decoder(motor_decoder_load_weights)
            self.set_learnable_motor_decoder(motor_decoder_learnable)

        if motor_decoder_helper_load_weights:
            self.load_weights_motor_decoder_helper(motor_decoder_helper_load_weights)
            self.set_learnable_motor_decoder_helper(motor_decoder_helper_learnable)

        if world_model_load_weights:
            self.load_weights_world_model(world_model_load_weights)
            self.set_learnable_world_model(world_model_learnable)

    @override(TorchModelV2)
    def get_initial_state(self):
        state = []
        return state

    def _reparameterize(self, mu, logvar):
        if self.latent_prior_noise:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        state_cnt = 0
        obs = input_dict["obs_flat"].float()
        z_body, z_task, state_cnt = self.forward_encoder(
            obs,
            state,
            seq_lens,
            state_cnt)
        logits, state_cnt = self.forward_decoder(
            z_body,
            z_task,
            state,
            seq_lens,
            state_cnt,
        )
        future_state = self.forward_world(obs, logits)

        val, state_cnt = self.forward_value_branch(
            obs,
            state,
            seq_lens,
            state_cnt)
        
        self._cur_body_encoder_variable = z_body
        self._cur_task_encoder_variable = z_task
        self._cur_value = val.squeeze(1)
        self._cur_future_state = future_state

        return logits, state

    def forward_encoder(self, obs, state, seq_lens, state_cnt):
        ''' Assume state==(state_body, state_task) '''
        
        if "body" in self._task_encoder_inputs and "task" in self._task_encoder_inputs:
            obs_task = obs[...,:]
        elif "body" in self._task_encoder_inputs:
            obs_task = obs[...,:self.dim_state_body]
        elif "task" in self._task_encoder_inputs:
            obs_task = obs[...,self.dim_state_body:]
        else:
            raise NotImplementedError

        obs_body = obs[...,:self.dim_state_body]
        z_body = obs_body

        z_task, state_cnt = forward_layer(
            obs_task,
            seq_lens,
            state, 
            state_cnt,
            self._task_encoder)

        if self._latent_prior_type in ["normal_zero_mean_one_std", "normal_state_mean_one_std"]: 
            mu = z_task[...,:self._task_encoder_output_dim]
            logvar = z_task[...,self._task_encoder_output_dim:]
            z_task = self._reparameterize(mu, logvar)
            self._cur_task_encoder_mu = mu
            self._cur_task_encoder_logvar = logvar
            if self._latent_prior_type == "normal_state_mean_one_std":
                z_prior, state_cnt = forward_layer(
                    obs_body,
                    seq_lens,
                    state, 
                    state_cnt,
                    self._latent_prior)
                self._cur_latent_prior_mu = z_prior
                self._cur_latent_prior_logvar = torch.zeros_like(z_prior)
        elif self._latent_prior_type in ["hypersphere_uniform"]:
            mu = torch.nn.functional.normalize(z_task)
            self._cur_task_encoder_mu = mu
            z_prior = torch.randn_like(mu)
            self._cur_latent_prior_mu = torch.nn.functional.normalize(z_prior)
        elif self._latent_prior_type == False:
            pass
        else:
            raise NotImplementedError

        return z_body, z_task, state_cnt

    def forward_decoder(self, z_body, z_task, state, seq_lens, state_cnt):
        z = []
        if "body" in self._motor_decoder_inputs:
            z.append(z_body)
        if "task" in self._motor_decoder_inputs:
            z.append(z_task)
        assert len(z) > 0
        z = torch.cat(z, axis=-1)
        
        logits = self._motor_decoder(z)

        if self._motor_decoder_helper:
            logits_add = self._motor_decoder_helper(z)
            logits[..., :self.dim_action] += self._motor_decoder_helper_range*logits_add
        
        return logits, state_cnt

    def forward_world(self, obs, logits):
        future_state = None
        if self._world_model:
            x = torch.cat([obs[...,:self.dim_state_body], logits[...,:self.dim_action]], axis=-1)
            future_state = self._world_model(x)
        return future_state

    def forward_value_branch(self, obs, state, seq_lens, state_cnt):
        val, state_cnt = forward_layer(
            obs,
            seq_lens,
            state, 
            state_cnt,
            self._value_branch)
        return val, state_cnt

    @override(TorchModelV2)
    def value_function(self):
        assert self._cur_value is not None, "must call forward() first"
        return self._cur_value

    def set_exploration_std(self, std):
        log_std = np.log(std)
        self._motor_decoder._model[-1].set_val(log_std)

    def task_encoder_variable(self):
        return self._cur_task_encoder_variable

    def body_encoder_variable(self):
        return self._cur_body_encoder_variable

    def save_weights(self, file):
        torch.save(self.state_dict(), file)

    def load_weights(self, file):
        self.load_state_dict(torch.load(file))
        self.eval()

    def save_weights_task_encoder(self, file):
        state_dict = {}
        state_dict['task_encoder'] = self._task_encoder.state_dict()
        torch.save(state_dict, file)

    def load_weights_task_encoder(self, file):
        state_dict = torch.load(file)
        self._task_encoder.load_state_dict(state_dict['task_encoder'])
        self._task_encoder.eval()

    def save_weights_motor_decoder(self, file):
        if self._motor_decoder:
            torch.save(self._motor_decoder.state_dict(), file)

    def save_weights_motor_decoder_helper(self, file):
        if self._motor_decoder_helper:
            torch.save(self._motor_decoder_helper.state_dict(), file)

    def load_weights_motor_decoder(self, file):
        if self._motor_decoder:
            ''' Ignore weights of log_std for valid exploration '''
            dict_weights_orig = self._motor_decoder.state_dict()
            dict_weights_loaded = torch.load(file)
            for key in dict_weights_loaded.keys():
                if 'log_std' in key:
                    dict_weights_loaded[key] = dict_weights_orig[key]
                    # print(dict_weights_orig[key])
            self._motor_decoder.load_state_dict(dict_weights_loaded)
            self._motor_decoder.eval()

    def load_weights_motor_decoder_helper(self, file):
        if self._motor_decoder_helper:
            self._motor_decoder_helper.load_state_dict(torch.load(file))
            self._motor_decoder_helper.eval()

    def save_weights_world_model(self, file):
        if self._world_model:
            torch.save(self._world_model.state_dict(), file)

    def load_weights_world_model(self, file):
        if self._world_model:
            self._world_model.load_state_dict(torch.load(file))
            self._world_model.eval()

    def save_weights_latent_prior(self, file):
        if self._latent_prior:
            torch.save(self._latent_prior.state_dict(), file)

    def load_weights_latent_prior(self, file):
        if self._latent_prior:
            self._latent_prior.load_state_dict(torch.load(file))
            self._latent_prior.eval()

    def set_learnable_task_encoder(self, learnable):
        if self._task_encoder:
            for name, param in self._task_encoder.named_parameters():
                param.requires_grad = learnable

    def set_learnable_motor_decoder(self, learnable, free_log_std=True):
        if self._motor_decoder:
            for name, param in self._motor_decoder.named_parameters():
                param.requires_grad = learnable
                if 'log_std' in name:
                    param.requires_grad = free_log_std 

    def set_learnable_motor_decoder_helper(self, learnable):
        if self._motor_decoder_helper:
            for name, param in self._motor_decoder_helper.named_parameters():
                param.requires_grad = learnable

    def set_learnable_world_model(self, learnable):
        if self._world_model:
            for name, param in self._world_model.named_parameters():
                param.requires_grad = learnable

ModelCatalog.register_custom_model("fcnn", FullyConnectedPolicy)
ModelCatalog.register_custom_model("physics_vae", PhysicsVAE)