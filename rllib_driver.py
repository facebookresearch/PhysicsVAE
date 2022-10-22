# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''
python rllib_driver.py --mode train --spec X --project_dir Y --local_dir Z
python rllib_driver.py --mode load --spec X --project_dir Y --checkpoint Z
'''

import os

import ray
from ray import tune

from ray.tune.registry import register_env
from ray.tune.logger import pretty_print

from collections import deque
import copy

import argparse
import yaml

ip_head = os.getenv('ip_head')
redis_password = os.getenv('redis_password')

print("ip_head:", ip_head)
print("redis_password:", redis_password)

def arg_parser():
    parser = argparse.ArgumentParser()
    ''' Mode for running an experiment '''
    parser.add_argument("--mode", required=True, choices=['train', 'load', 'gen_expert_demo'])
    ''' Specification file of the expriment '''
    parser.add_argument("--spec", required=True, action='append')
    '''  '''
    parser.add_argument("--checkpoint", action='append', default=[])
    '''  '''
    parser.add_argument("--num_workers", type=int, default=None)
    '''  '''
    parser.add_argument("--num_cpus", type=int, default=1)
    '''  '''
    parser.add_argument("--num_gpus", type=int, default=0)
    '''  '''
    parser.add_argument("--num_envs_per_worker", type=int, default=None)
    '''  '''
    parser.add_argument("--num_cpus_per_worker", type=int, default=None)
    '''  '''
    parser.add_argument("--num_gpus_per_worker", type=int, default=None)
    ''' Directory where the environment and related files are stored '''
    parser.add_argument("--project_dir", type=str, default=None)
    ''' Directory where intermediate results are saved '''
    parser.add_argument("--local_dir", type=str, default=None)
    ''' Verbose '''
    parser.add_argument("--verbose", action='store_true')
    '''  '''
    parser.add_argument("--ip_head", type=str, default=None)
    '''  '''
    parser.add_argument("--password", type=str, default=None)
    '''  '''
    parser.add_argument("--width", type=int, default=1280)
    '''  '''
    parser.add_argument("--height", type=int, default=720)
    '''  '''
    parser.add_argument("--bgcolor", type=str, default="black")
    '''  '''
    parser.add_argument("--renderer", choices=['inhouse', 'bullet_native'], default='inhouse')
    '''  '''
    parser.add_argument("--temp_dir", type=str, default='/tmp/ray/')
    '''  '''
    parser.add_argument("--kill_previous_run", action='store_true')

    return parser

if __name__ == "__main__":

    args = arg_parser().parse_args()

    assert len(args.spec) > 0

    if args.kill_previous_run:
        print('>> Shutdown previous run if exist')
        ray.shutdown()

    if ip_head is not None:
        print('>> Trying to initialize Ray as HEAD...')
        # tmp_dir = os.path.join(spec['local_dir'], os.path.join('tmp/', spec['name']))
        if redis_password:
            ray.init(
                address=ip_head, 
                _redis_password=redis_password, 
                _temp_dir=args.temp_dir)
        else:
            ray.init(address=ip_head, 
                _temp_dir=args.temp_dir)
        print('>> Ray was initialized as HEAD')
    else:
        assert args.num_cpus is not None
        assert args.num_gpus is not None
        print('>> Trying to initialize Ray as CLIENT...')
        print('num_cpus:', args.num_cpus)
        print('num_gpus:', args.num_gpus)
        print('redis_password:', redis_password)
        if redis_password:
            ray.init(
                num_cpus=args.num_cpus, 
                num_gpus=args.num_gpus, 
                _redis_password=redis_password, 
                _temp_dir=args.temp_dir)
        else:
            ray.init(
                num_cpus=args.num_cpus,
                num_gpus=args.num_gpus, 
                _temp_dir=args.temp_dir)
        print('>> Ray was initialized as CLIENT')

    config_list = []
    spec_list = []
    for spec_file in args.spec:
        with open(spec_file) as f:
            spec = yaml.load(f, Loader=yaml.FullLoader)
        config = spec['config']

        '''
        Register environment to learn according to the input specification file
        '''

        if config['env'] == "HumanoidImitation":
            from envs import rllib_env_imitation as env_module
        else:
            raise NotImplementedError("Unknown Environment")

        register_env(config['env'], lambda config: env_module.env_cls(config))

        '''
        Register custom model to use if it exists
        '''

        framework = config.get('framework')

        if config.get('model'):
            custom_model = config.get('model').get('custom_model')
            if custom_model:
                if framework=="torch":
                    import rllib_model_torch
                else:
                    raise NotImplementedError

        '''
        Validate configurations and overide values by arguments
        '''

        if args.local_dir is not None:
            spec.update({'local_dir': args.local_dir})
        
        if args.project_dir is not None:
            assert os.path.exists(args.project_dir)
            config['env_config']['project_dir'] = args.project_dir
            if 'base_env_config' in config['env_config']:
                config['env_config']['base_env_config']['project_dir'] =\
                    args.project_dir
        
        if config['model'].get('custom_model_config'):
            config['model']['custom_model_config'].update(
                {'project_dir': config['env_config']['project_dir']})

        if args.verbose:
            config['env_config'].update({'verbose': args.verbose})
            if 'base_env_config' in config['env_config']:
                config['env_config']['base_env_config'].update(
                    {'verbose': args.verbose})
        
        if args.num_workers is not None:
            config.update({'num_workers': args.num_workers})
        
        if args.num_gpus is not None:
            config.update({'num_gpus': args.num_gpus})

        if args.num_envs_per_worker:
            config.update({'num_envs_per_worker': args.num_envs_per_worker})

        if args.num_cpus_per_worker:
            config.update({'num_cpus_per_worker': args.num_cpus_per_worker})

        if args.num_gpus_per_worker:
            config.update({'num_gpus_per_worker': args.num_gpus_per_worker})

        if args.mode == "train":
            if not os.path.exists(spec['local_dir']):
                raise Exception(
                    "The directory does not exist: %s"%spec['local_dir'])

        config_override = env_module.config_override(spec)
        config.update(config_override)

        def adjust_config(config, alg):
            rollout_fragment_length = config.get('rollout_fragment_length')
            num_workers = config.get('num_workers')
            num_envs_per_worker = config.get('num_envs_per_worker')
            train_batch_size = config.get('train_batch_size')

            ''' 
            Set rollout_fragment_length value so that
            workers can genertate train_batch_size tuples correctly
            '''
            rollout_fragment_length = \
                max(train_batch_size // (num_workers * num_envs_per_worker), 1)

            config['rollout_fragment_length'] = rollout_fragment_length
            
            if alg in ['DDPPO']:
                config['train_batch_size'] = -1

        adjust_config(config, spec['run'])

        spec_list.append(spec)
        config_list.append(config)
    
    if args.mode == "load" or args.mode == "gen_expert_demo":
        def adjust_config_for_loading(config):
            config["num_workers"] = 1
            config['num_envs_per_worker'] = 1
            config['num_cpus_per_worker'] = 1
            config['num_gpus_per_worker'] = 0
            config['remote_worker_envs'] = False

        def load_trainer_cls(spec):
            if spec["run"] == "PPO":
                from ray.rllib.agents.ppo import PPOTrainer as Trainer
            elif spec["run"] == "DDPPO":
                from ray.rllib.agents.ppo import DDPPOTrainer as Trainer
            else:
                raise NotImplementedError
            return Trainer

        adjust_config_for_loading(config_list[0])

        trainers = []
        
        trainer_cls = load_trainer_cls(spec_list[0])
        trainer = trainer_cls(env=env_module.env_cls, config=config_list[0])
        if len(args.checkpoint) > 0:
            trainer.restore(args.checkpoint[0])
        trainers.append(trainer)

        if args.mode == "load":
            env_module.rm.initialize()
            if args.bgcolor == "black":
                bgcolor = [0.0, 0.0, 0.0, 1.0]
            elif args.bgcolor == "white":
                bgcolor = [1.0, 1.0, 1.0, 1.0]
            elif args.bgcolor == "clear":
                bgcolor = [0.0, 0.0, 0.0, 0.0]
            else:
                raise NotImplementedError
            env = env_module.env_cls(config_list[0]['env_config'])
            renderer = env_module.EnvRenderer(
                trainers=trainers, 
                env=env, 
                cam=env_module.default_cam(env), 
                renderer=args.renderer, 
                size=(args.width, args.height), 
                bgcolor=bgcolor, 
                config=config_list[0])
            renderer.run()
        elif args.mode == "gen_expert_demo":
            if config_list[0]["env_config"].get("lazy_creation"):
                config_list[0]["env_config"]["lazy_creation"] = False
            env = env_module.env_cls(config_list[0]['env_config'])
            env_module.gen_state_action_pairs(trainer, env)
    else:
        spec = spec_list[0]
        config = config_list[0]
        resume = False
        if len(args.checkpoint) > 0:
            checkpoint = args.checkpoint[0]
        else:
            checkpoint = None
        if checkpoint is None and os.path.exists(
            os.path.join(spec['local_dir'], spec['name'])
        ):
            resume = "ERRORED_ONLY"
        if spec['run'] in ["PPO", "DDPPO"]:
            tune.run(
                spec['run'],
                name=spec['name'],
                stop=spec['stop'],
                local_dir=spec['local_dir'],
                checkpoint_freq=spec['checkpoint_freq'],
                checkpoint_at_end=spec['checkpoint_at_end'],
                config=config,
                resume=resume,
                restore=checkpoint,
                reuse_actors=spec.get('reuse_actors', True),
                raise_on_failed_trial=False,
            )
        else:
            raise NotImplementedError
