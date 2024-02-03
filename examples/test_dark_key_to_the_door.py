from argparse import ArgumentParser

import wandb

import PSBL
from PSBL.envs.builtin.gym_envs import GymEnv
from PSBL.envs.builtin.room_key_door import RoomKeyDoor
from example_utils import *
import os
os.environ["WANDB_MODE"] = "offline"

def add_cli(parser):
    parser.add_argument("--meta_horizon", type=int, default=500)
    parser.add_argument("--room_size", type=int, default=9)
    parser.add_argument("--episode_length", type=int, default=50)
    return parser


if __name__ == "__main__":
    parser = ArgumentParser()
    add_common_cli(parser)
    add_cli(parser)
    args = parser.parse_args()


    args.run_name="test"
    args.gpu=0
    args.parallel_actors=64
    mine_flag=True
    args.grads_per_epoch=2000
    args.batch_size=64
    args.slow_inference=True
    args.slow_inference=1
    args.memory_len=500
    args.agent_network=1



    group_name = f"{args.run_name}_dark_key_door"
    for trial in range(args.trials):
        run_name = group_name + f"_trial_{trial}"
       
        make_train_env = lambda: GymEnv(
            gym_env=RoomKeyDoor(
                size=args.room_size, max_episode_steps=args.episode_length
            ),
            env_name="Dark-Key-To-Door",
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
            group_name=group_name,
            run_name=run_name,
            val_timesteps_per_epoch=2000,
            meta_horizon=args.meta_horizon,
            episode_length=args.episode_length,
            



    
        )
        model_path='examples/model/DarkKeyDoor_policy.pt'
        
        experiment.start()
        experiment.load_policy(model_path)
    
        experiment.evaluate_test_episode(make_train_env, timesteps=20_000, render=False,episode_length=args.episode_length+1,traj_length=800)
        wandb.finish()
