# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import ray
from ray import tune

import pickle

import torch_models
import rllib_model_torch as policy_models

import gym
from gym.spaces import Box

import argparse
import copy

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_iter_world_model", type=int, default=0)
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--num_cpus", type=int, default=1)
    parser.add_argument("--num_gpus", type=int, default=0)
    parser.add_argument("--data_train", action="append", required=True, type=str, default=None)
    parser.add_argument("--data_test", action="append", type=str, default=None)
    parser.add_argument("--num_data", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--lr_schedule", type=str, default="step")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--checkpoint_freq", type=int, default=100)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--cluster", action='store_true')
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--local_dir", type=str, default='~/ray_results')
    parser.add_argument("--world_model", type=str, default=None)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--vae_kl_coeff", type=float, action='append', default=[1.0])
    parser.add_argument("--vae_cycle_coeff", type=float, action='append', default=[1e-3])
    parser.add_argument("--latent_prior_type", type=str, action='append', default=['normal_zero_mean_one_std'])

    return parser

'''
The prepared dataset should be a dictionary that looks like
{
    dim_action: 
        The dimension of action
    dim_state: 
        The dimension of state
    dim_state_body: 
        The dimension of proprioceptive state
    dim_state_task: 
        The dimension of task (goal) state
    exp_std: 
        std of Gaussian policy used for generating episodes. 
        This is the main source why each episode is different from each other
        even if they be run with the same referece motion
    iter_per_episode:
        How many episodes are generated for the same reference motion
    episodes: 
        A list of episodes [ep_1, ep_2, ..., ep_N]
}
Each episode is a dictionary that looks like
{
    time:
        A list of elapsed times [t_1, t_2, ..., t_T]
    state:
        A list of states [s_1, s_2, ..., s_T]
    state_body:
        A list of proprioceptive states [sb_1, sb_2, ..., sb_T]
    state_task:
        A list of task (goal) states [sg_1, sg_2, ..., sg_T]
    action:
        A list of actions [a_1, a_2, ..., a_T]
    reward:
        A list of rewards [r_1, r_2, ..., r_T]
}
'''

def merge_dataset(files):
    '''
    Merge two different datasets
    '''
    data_all = None
    for i, file in enumerate(files):
        with open(file, "rb") as f:
            ''' Load saved data '''
            data = pickle.load(f)
            print(file, 'is loaded')
            if i==0:
                data_all = data
            else:
                assert data_all['iter_per_episode'] == data['iter_per_episode']
                assert data_all['dim_state'] == data['dim_state']
                assert data_all['dim_state_body'] == data['dim_state_body']
                assert data_all['dim_state_task'] == data['dim_state_task']
                assert data_all['dim_action'] == data['dim_action']
                assert data_all['exp_std'] == data['exp_std']
                data_all['episodes'] = data_all['episodes'] + data['episodes']
    return data_all


def load_dataset_for_PhysicsVAE(
    files, 
    num_samples=None, 
    lookahead=1, 
    cond="abs", 
    use_a_gt=False):

    assert files
    assert len(files) > 0

    data = merge_dataset(files)
    episodes = data["episodes"]
    X, Y = [], []

    assert lookahead >= 1

    for ep in episodes:
        num_tuples = len(ep["time"])
        assert num_tuples >= lookahead
        for i in range(num_tuples-lookahead):
            if num_samples is not None and len(X) >= num_samples:
                break
            x = []
            y = []
            for j in range(lookahead):
                state_body_t1 = ep["state_body"][i+j]
                state_body_t2 = ep["state_body"][i+j+1]
                if use_a_gt:
                    action = ep["action_gt"][i+j]
                else:
                    action = ep["action"][i+j]
                if cond == "abs":
                    x.append(np.hstack([state_body_t1, state_body_t2]))
                elif cond == "rel":
                    x.append(np.hstack([state_body_t1, state_body_t2-state_body_t1]))
                else:
                    raise NotImplementedError
                y.append(action)
            X.append(np.vstack(x))
            Y.append(np.vstack(y))

    print("------------------Data Loaded------------------")
    print("File:", files)
    print("Num Episodes:", len(episodes))
    print("Num Transitions (Tuples):", len(X))
    print("-----------------------------------------------")
    return torch_models.DatasetBase(
            np.array(X), np.array(Y), normalize_x=False, normalize_y=False)

def create_model(config):
    model_config = config["model"]
    obs_space = model_config["custom_model_config"]["observation_space"]
    action_space = model_config["custom_model_config"]["action_space"]
    return policy_models.PhysicsVAE(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=2*action_space.shape[0],
        model_config=model_config,
        name="physics_vae"
        )

MODEL_CONFIG = copy.deepcopy(policy_models.PhysicsVAE.DEFAULT_CONFIG)

def gen_layers(width, depth, out_size="output", act_hidden="relu", act_out="linear", add_softmax=False):
    assert depth > 0 and width > 0
    layers = []
    for i in range(depth):
        layers.append(
            {"type": "fc", "hidden_size": width, "activation": act_hidden, "init_weight": {"name": "normc", "std": 1.0}}
        )
    layers.append(
        {"type": "fc", "hidden_size": out_size, "activation": act_out, "init_weight": {"name": "normc", "std": 0.01}}
    )
    if add_softmax:
        layers.append({"type": "softmax"})
    return layers

def get_trainer_config(args):

    assert args.max_iter_world_model <= args.max_iter

    '''
    For PhysicsVAE model, we use 'state_body' at time t+1 (sb_{t+1}) 
    as our task/goal state at time t (sg_{t}). So, when constructing the model, 
    the entire state should be s_t = (sb_{t},sg_{t}) = (sb_{t},sb_{t+1}).
    '''
    def inspect_dataset(file):
        with open(file, "rb") as f:
            ''' Load saved data '''
            data = pickle.load(f)
            ep0 = data["episodes"][0]
            dim_state_body = len(ep0["state_body"][0])
            dim_action = len(ep0["action"][0])
            return 2*dim_state_body, dim_state_body, dim_state_body, dim_action
        raise Exception('File error')

    dim_state, dim_state_body, dim_state_task, dim_action = \
        inspect_dataset(args.data_train[0])

    ob_scale = 1000.0
    ac_scale = 3.0
    obs_space = \
        Box(low=-ob_scale*np.ones(dim_state),
            high=ob_scale*np.ones(dim_state),
            dtype=np.float64)
    obs_space_body = \
        Box(low=-ob_scale*np.ones(dim_state_body),
            high=ob_scale*np.ones(dim_state_body),
            dtype=np.float64)
    obs_space_task = \
        Box(low=-ob_scale*np.ones(dim_state_task),
            high=ob_scale*np.ones(dim_state_task),
            dtype=np.float64)
    action_space = \
        Box(low=-ac_scale*np.ones(dim_action),
            high=ac_scale*np.ones(dim_action),
            dtype=np.float64)

    model_config = {
        "custom_model": "physics_vae",
        "custom_model_config": MODEL_CONFIG.copy(),
    }
    custom_model_config = model_config["custom_model_config"]
    custom_model_config["observation_space"] = obs_space
    custom_model_config["observation_space_body"] = obs_space_body
    custom_model_config["observation_space_task"] = obs_space_task
    custom_model_config["action_space"] = action_space

    custom_model_config["world_model_load_weights"] = args.world_model

    trainer_config = {
        "max_iter_world_model": args.max_iter_world_model,
        "model": model_config,
        "lr": args.lr,
        "lr_schedule_params": {"step_size": 50, "gamma": 0.70},
        "lr_schedule": args.lr_schedule,
        "weight_decay": 0.0,
        "dataset_train": args.data_train,
        "dataset_test": args.data_test,
        "use_gpu": False,
        "loss": "MSE",
        "loss_test": "MSE",
        "batch_size": args.batch_size,
        "suffle_data": True,
        
        ## VAE related parameters
        "latent_dim": args.latent_dim,
        "latent_prior_type": tune.grid_search(args.latent_prior_type),

        "act_fn": "relu",
        
        ## Motor Decoder
        "MD_width": tune.grid_search([512]),
        "MD_depth": tune.grid_search([3]),

        ## Task Encoder
        "TE_width": tune.grid_search([256]),
        "TE_depth": tune.grid_search([2]),

        ## World Model
        "lookahead": 1,
        "world_model_width": tune.grid_search([1024]),
        "world_model_depth": tune.grid_search([2]),

        # Learning Coefficients
        "vae_kl_coeff": tune.grid_search(args.vae_kl_coeff),
        "motor_decoder_a_rec_coeff": 1.0,
        "world_model_s_rec_coeff": 0.0,
        "vae_cycle_coeff": tune.grid_search(args.vae_cycle_coeff),
    }

    return trainer_config

def update_model_config(trainer_config):
    model_config = trainer_config["model"]["custom_model_config"]
    model_config["task_encoder_output_dim"] = trainer_config.get("latent_dim")
    model_config["task_encoder_layers"] = \
        gen_layers(
            width=trainer_config.get("TE_width"),
            depth=trainer_config.get("TE_depth"),
            act_hidden=trainer_config.get("act_fn"),
        )
    model_config["motor_decoder_layers"] = \
        gen_layers(
            width=trainer_config.get("MD_width"),
            depth=trainer_config.get("MD_depth"),
            act_hidden=trainer_config.get("act_fn"),
        )
    model_config["latent_prior_type"] = trainer_config.get("latent_prior_type")
    model_config["world_model_layers"] = \
        gen_layers(
            width=trainer_config.get("world_model_width"),
            depth=trainer_config.get("world_model_depth"),
            act_hidden=trainer_config.get("act_fn"),
        )

class TrainModel(torch_models.TrainModel):
    def setup(self, config):        
        update_model_config(config)

        self.config = config
        self.max_iter_world_model = config.get("max_iter_world_model")
        
        self.latent_prior_type = config.get("latent_prior_type")
        
        self.lookahead = config.get("lookahead")

        super().setup(config)

        self.model.set_learnable_task_encoder(False)
        self.model.set_learnable_motor_decoder(False)
        self.model.set_learnable_world_model(True)
        self.read_loss_fn_coeff(world=True)

    def read_loss_fn_coeff(self, world):
        self.vae_kl_coeff = 0.0 if world else self.config.get("vae_kl_coeff")
        self.a_rec_coeff = 0.0 if world else self.config.get("motor_decoder_a_rec_coeff")
        self.s_rec_coeff = 1.0 if world else self.config.get("world_model_s_rec_coeff")
        self.vae_cycle_coeff = 0.0 if world else self.config.get("vae_cycle_coeff")

    def load_dataset(self, file):
        return load_dataset_for_PhysicsVAE(
            file, num_samples=args.num_data, lookahead=self.lookahead)

    def step(self):
        if self.iter == self.max_iter_world_model:
            '''
            When the current iteration reaches to the max_iter_world_model,
            end-to-end learning of the model starts
            '''
            self.model.set_learnable_task_encoder(True)
            self.model.set_learnable_motor_decoder(True)
            self.model.set_learnable_world_model(False)
            self.read_loss_fn_coeff(world=False)
        return super().step()
    
    def create_model(self, config):
        return create_model(config)
    
    def compute_model(self, x):
        logits, _ = self.model(
            input_dict={"obs": x, "obs_flat": x}, state=None, seq_lens=None)
        return logits[...,:logits.shape[1]//2]
    
    def compute_loss(self, y, x):        
        loss_a = loss_kl = loss_s = loss_cyc = 0.0

        # State at time t
        s1 = x[..., 0, :self.model.dim_state_body]

        for t in range(self.lookahead):

            x_t_gt = x[..., t, :].squeeze()
            y_t_gt = y[..., t, :].squeeze()

            # State at time t
            s1_gt = x_t_gt[..., :self.model.dim_state_body]
            # State at time t+1
            s2_gt = x_t_gt[..., self.model.dim_state_body:]

            x_t = torch.cat([s1, s2_gt], axis=-1)
            y_t = self.compute_model(x_t)

            # Action (from Motor Decoder) Reconstruction Loss
            if self.a_rec_coeff > 0.0:
                loss_a += self.loss_fn(y_t_gt, y_t)
                if self.latent_prior_type and self.vae_kl_coeff > 0.0:
                    if self.latent_prior_type == "normal_zero_mean_one_std":
                        # KL-loss
                        mu = self.model._cur_task_encoder_mu
                        logvar = self.model._cur_task_encoder_logvar
                        kl_div = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),dim=1),dim=0)
                        loss_kl += kl_div
                    elif self.latent_prior_type == "normal_state_mean_one_std":
                        # KL-loss
                        mu1 = self.model._cur_task_encoder_mu
                        logvar1 = self.model._cur_task_encoder_logvar
                        std1 = torch.exp(0.5 * logvar1)
                        mu2 = self.model._cur_vae_prior_mu
                        logvar2 = self.model._cur_vae_prior_logvar
                        std2 = torch.exp(0.5 * logvar2)
                        kl_div = 0
                        for i in range(mu1.shape[0]):
                            p = torch.distributions.Normal(mu1[i,:],std1[i,:])
                            q = torch.distributions.Normal(mu2[i,:],std2[i,:])
                            kl_div += torch.distributions.kl_divergence(p, q)
                        loss_kl += kl_div.mean()
                    elif self.latent_prior_type == "hypersphere_uniform":
                        mu1 = self.model._cur_task_encoder_mu
                        mu2 = self.model._cur_vae_prior_mu
                        loss_kl += (mu1*mu2).sum(-1).mean()
                    else:
                        raise NotImplementedError

            # State Reconstruction (from World Model) Loss
            if self.s_rec_coeff > 0:
                s2_pred = self.model.forward_world(s1, y_t_gt)
                loss_s += self.loss_fn(s2_gt, s2_pred)

            # VAE Cycle Consistency Loss
            if self.vae_cycle_coeff > 0:
                s2_pred = self.model._cur_future_state
                loss_cyc += self.loss_fn(s2_gt, s2_pred)

            s1 = self.model._cur_future_state
        
        if self.lookahead > 1:
            N = float(self.lookahead)
            loss_a /= N
            loss_kl /= N
            loss_s /= N
            loss_cyc /= N

        total_loss = \
            self.a_rec_coeff * loss_a + \
            self.vae_kl_coeff * loss_kl + \
            self.s_rec_coeff * loss_s + \
            self.vae_cycle_coeff * loss_cyc
        return total_loss
    
    def compute_test_loss(self, y, x):
        return self.compute_loss(y, x)

    def save_checkpoint(self, checkpoint_dir):
        checkpoint_path = super().save_checkpoint(checkpoint_dir)
        ''' Save the entire model '''
        checkpoint = os.path.join(checkpoint_dir, "model.pt")
        self.model.save_weights(checkpoint)
        print("Saved:", checkpoint)
        ''' Save each component '''
        if self.model._task_encoder:
            checkpoint = os.path.join(
                checkpoint_dir, "task_encoder.pt")
            self.model.save_weights_task_encoder(checkpoint)
            print("Saved:", checkpoint)
        if self.model._motor_decoder:
            checkpoint = os.path.join(
                checkpoint_dir, "motor_decoder.pt")
            self.model.save_weights_motor_decoder(checkpoint)
            print("Saved:", checkpoint)
        if self.model._world_model:
            checkpoint = os.path.join(
                checkpoint_dir, "world_model.pt")
            self.model.save_weights_world_model(checkpoint)
            print("Saved:", checkpoint)
        if self.model._latent_prior:
            checkpoint = os.path.join(
                checkpoint_dir, "latent_prior.pt")
            self.model.save_weights_latent_prior(checkpoint)
            print("Saved:", checkpoint)
        return checkpoint_path

if __name__ == "__main__":

    args = arg_parser().parse_args()

    ''' Prepare data and initialize ray '''

    if args.cluster:
        print('>> Trying to initialize Ray')
        ray.init(num_cpus=args.num_cpus, num_gpus=args.num_gpus)
        print('>> Ray was initialized')

    trainer_config = get_trainer_config(args)

    if args.checkpoint is None:
        ''' Train a Model '''
        analysis = tune.run(
            TrainModel,
            # scheduler=sched,
            stop={
                # "mean_accuracy": 0.95,
                "training_iteration": args.max_iter,
            },
            resources_per_trial={
                "cpu": 1,
                "gpu": 1 if args.num_gpus > 0 else 0,
            },
            num_samples=1,
            checkpoint_freq=args.checkpoint_freq,
            checkpoint_at_end=True,
            config=trainer_config,
            reuse_actors=True,
            local_dir=args.local_dir,
            resume=args.resume,
            name=args.name)

        logdir = analysis.get_best_logdir(
            metric='training_iteration', mode='max')
        checkpoint = analysis.get_best_checkpoint(
            logdir, metric='training_iteration', mode='max')
    else:
        checkpoint = args.checkpoint

    if args.output is not None:
        trainer = torch_models.TrainModel(trainer_config)
        trainer.restore(checkpoint)
        torch.save(
            trainer.model.state_dict(), 
            args.output,
        )
        print('Model Saved:', args.output)

    if args.cluster:
        ray.shutdown()