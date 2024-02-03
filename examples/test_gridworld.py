from argparse import ArgumentParser

import wandb

import PSBL
from PSBL.envs.builtin.gym_envs import GymEnv
from PSBL.envs.builtin.room_key_door import RoomKeyDoor
from example_utils import *
from PSBL.envs.builtin.gridworld import GridNavi
import os
os.environ["WANDB_MODE"] = "offline"


def add_cli(parser):
    parser.add_argument("--meta_horizon", type=int, default=300)
    parser.add_argument("--room_size", type=int, default=9)
    parser.add_argument("--episode_length", type=int, default=15)
    return parser


if __name__ == "__main__":
    parser = ArgumentParser()
    add_common_cli(parser)
    add_cli(parser)
    args = parser.parse_args()

    args.run_name="test"
    args.gpu=0
    args.parallel_actors=64
    args.grads_per_epoch=2000
    args.batch_size=32
    args.timesteps_per_epoch=1000
    args.init_learning_rate=5e-5
    args.weighted_sample=0
    args.slow_inference=1
    args.memory_len=150
    args.agent_network=1
    mine_flag=True
    args.on_line=mine_flag
   

    group_name = f"{args.run_name}_gridworld"
    for trial in range(args.trials):
        run_name = group_name + f"_trial_{trial}"
        """
        Meta-RL is the same as any other environment except that we reset
        the task over and over again until a fixed time limit. The easiest
        way to do this in PSBL is to design/wrap the meta-RL gym Env so that
        `env.reset()` *picks a new task* ("hard reset") but create kwargs to 
        reset to the *same* task ("soft reset"). In this example `RoomKeyDoor.reset()` 
        randomly generates a new environment while `RoomKeyDoor.reset(new_task=False) 
        resets the agent to the same environment.

        For meta-RL by *episodes* (k-shot) instead of
        timesteps, see the metaworld example.
        """
        make_train_env = lambda: GymEnv(
            gym_env=GridNavi(num_cells=5,
                max_episode_steps=15,
            ),
            env_name="gridworld",
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
            wandb_project="gridworld",
            init_learning_rate=args.init_learning_rate,
            weighted_sample=args.weighted_sample,
            memory_len=args.memory_len,
            agent_network=args.agent_network,


    
        )

        
        
        model_path='examples/model/Gridworld_policy.pt'
        experiment.start()
        experiment.load_policy(model_path)
    
        experiment.evaluate_test_episode_grid(make_train_env, timesteps=20_000, render=False,episode_length=args.episode_length+1,traj_length=900)
        wandb.finish()
