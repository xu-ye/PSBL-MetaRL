from argparse import ArgumentParser

import wandb

import PSBL
from PSBL.envs.builtin.gym_envs import GymEnv
from PSBL.envs.builtin.room_key_door import RoomKeyDoor
from example_utils import *
from PSBL.envs.builtin.half_cheetah_vel import HalfCheetahVelEnv
import os
os.environ["WANDB_MODE"] = "offline"

def add_cli(parser):
    parser.add_argument("--meta_horizon", type=int, default=600)
    parser.add_argument("--room_size", type=int, default=9)
    parser.add_argument("--episode_length", type=int, default=200)
    return parser


if __name__ == "__main__":
    parser = ArgumentParser()
    add_common_cli(parser)
    add_cli(parser)
    args = parser.parse_args()

    args.run_name="half_cheetah"
    args.parallel_actors=64
    args.grads_per_epoch=2000
    args.batch_size=32
    args.timesteps_per_epoch=2000
    args.memory_layers=3
    args.slow_inference=1
    args.memory_len=600
    args.agent_network=1
    args.episode_emb=0
    mine_flag=True
    args.on_line=mine_flag


    

    group_name = f"{args.run_name}_half_cheetah_vel"
    for trial in range(args.trials):
        run_name = group_name + f"_trial_{trial}"
        
        make_train_env = lambda: GymEnv(
            gym_env=HalfCheetahVelEnv(
                max_episode_steps=args.episode_length
            ),
            env_name="Half_cheetah_vel",
            horizon=args.meta_horizon,
            zero_shot=False,
            # env.reset() is called between rollouts (new tasks), while
            # env.reset(**soft_reset_kwargs) is called within meta-RL rollouts
            # (same task).
            soft_reset_kwargs={"new_task": False},
        )

        experiment = create_experiment_from_cli(
            args,
            make_train_env=make_train_env,
            make_val_env=make_train_env,
            max_seq_len=args.meta_horizon,
            traj_save_len=args.meta_horizon,
            #group_name="half-cheetch",
            group_name=group_name,
            run_name=run_name,
            val_timesteps_per_epoch=2000,
            meta_horizon=args.meta_horizon,
            episode_length=args.episode_length,
            on_line=args.on_line,
            wandb_project="half-cheetch",
            #weighted_sample=args.weighted_sample,
            memory_len=args.memory_len,
            agent_network=args.agent_network,
            episode_emb=args.episode_emb,


    
        )
       
        model_path='examples/model/Halfcheetah_policy.pt'
        experiment.start()
        experiment.load_policy(model_path)
        experiment.evaluate_test_episode(make_train_env, timesteps=20_000, render=False,episode_length=args.episode_length+1,traj_length=1600)
        
        wandb.finish()
