from functools import partial
from argparse import ArgumentParser
from typing import Callable
import torch

import gin

import PSBL


def add_common_cli(parser: ArgumentParser) -> ArgumentParser:
    # extra gin configs
    parser.add_argument(
        "--configs",
        type=str,
        nargs="*",
        help="Extra `.gin` configuration files. These settings are usually added last in the examples and would overwrite the script's defaults.",
    )
    # basics
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--no_log",
        action="store_true",
        help="Turn off wandb logging (usually for debugging).",
    )
    parser.add_argument(
        "--ckpt",
        type=int,
        default=None,
        help="Start training from an epoch checkpoint saved in a buffer with the same `--run_name`",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="test",
        #required=True,
        help="Give the run a name. Used for logging and the disk replay buffer. Experiments with the same run_name share the same replay buffer, but log separately.",
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument(
        "--buffer_dir",
        type=str,
        default="/home/xty/code/TransformerOnline/buffer",
        #required=True,
        help="Path to disk location where replay buffer (and checkpoints) will be stored. Should probably be somewhere with lots of space...",
    )
    # trajectory encoder
    parser.add_argument(
        "--traj_encoder", choices=["ff", "transformer", "rnn"], default="transformer"
    )
    parser.add_argument(
        "--memory_size",
        type=int,
        default=256,
        help="Model/token dimension for a Transformer; hidden state size for an RNN.",
    )
    parser.add_argument(
        "--memory_layers",
        type=int,
        default=3,
        help="Number of layers in the sequence model.",
    )
    # main learning schedule
    parser.add_argument(
        "--grads_per_epoch",
        type=int,
        default=1000,
        help="Gradient updates per training epoch.",
    )
    parser.add_argument(
        "--timesteps_per_epoch",
        type=int,
        default=1000,
        help="Timesteps of environment interaction per epoch. The update:data ratio is defined by `grads_per_epoch / (timesteps_per_epoch * parallel_actors)`.",
    )
    parser.add_argument(
        "--val_interval",
        type=int,
        default=2,
        help="How often (in epochs) to evaluate the agent on validation envs.",
    )
    parser.add_argument(
        "--ckpt_interval",
        type=int,
        default=20,
        help="How often (in epochs) to save an agent checkpoint.",
    )
    parser.add_argument(
        "--parallel_actors",
        type=int,
        default=12,
        help="Number of parallel environments (applies to training, validation, and testing).",
    )
    parser.add_argument(
        "--dset_max_size",
        type=int,
        default=20_000,
        help="Maximum size of the replay buffer (measured in trajectories, not timesteps).",
    )
    parser.add_argument(
        "--start_learning_after_epoch",
        type=int,
        default=0,
        help="Skip learning updates for this many epochs at the beginning of training (if worried about overfitting to a small dataset)",
    )
    parser.add_argument(
        "--slow_inference",
        action="store_true",
        help="Turn OFF fast-inference mode (key-value caching for Transformer, hidden state caching for RNN)",
    )
    parser.add_argument(
        "--half_precision",
        action="store_true",
        help="Train in bfloat16 half precision",
    )
    parser.add_argument(
        "--dloader_workers",
        type=int,
        default=8,
    )
    return parser


"""
Gin convenience functions.

Switch between the most common configurations without needing `.gin` config files.
"""




def naive(config: dict):
    config.update(
        {
            "PSBL.nets.traj_encoders.TformerTrajEncoder.activation": "gelu",
            "PSBL.nets.actor_critic.NCritics.activation": "relu",
            "PSBL.nets.actor_critic.Actor.activation": "relu",
            "PSBL.nets.tstep_encoders.FFTstepEncoder.activation": "relu",
            "PSBL.nets.tstep_encoders.CNNTstepEncoder.activation": "relu",
            "PSBL.nets.transformer.TransformerLayer.normformer_norms": False,
            "PSBL.nets.transformer.TransformerLayer.sigma_reparam": False,
            "PSBL.nets.transformer.AttentionLayer.sigma_reparam": False,
            "PSBL.nets.transformer.AttentionLayer.head_scaling": False,
            "PSBL.agent.Agent.num_critics": 2,
            "PSBL.agent.Agent.gamma": 0.99,
            "PSBL.agent.Agent.use_multigamma": False,
        }
    )
    return config


def use_config(custom_params: dict, gin_configs: list[str] | None = None):
    """
    Bind all the gin parameters from real .gin configs (which the examples avoid using)
    and regular dictionaries.

    Use before training begins.
    """
    for param, val in custom_params.items():
        gin.bind_parameter(param, val)
    # override defaults with custom gin config files
    if gin_configs is not None:
        for config in gin_configs:
            gin.parse_config_file(config)
    gin.finalize()


def create_experiment_from_cli(
    command_line_args,
    make_train_env: Callable,
    make_val_env: Callable,
    max_seq_len: int,
    traj_save_len: int,
    group_name: str,
    run_name: str,
    **extra_experiment_kwargs,
):
    cli = command_line_args

    experiment = PSBL.Experiment(
        make_train_env=make_train_env,
        make_val_env=make_val_env,
        max_seq_len=max_seq_len,
        traj_save_len=traj_save_len,
        dset_max_size=cli.dset_max_size,
        run_name=run_name,
        dset_name=run_name,
        gpu=cli.gpu,
        dset_root=cli.buffer_dir,
        dloader_workers=cli.dloader_workers,
        log_to_wandb=not cli.no_log,
        wandb_group_name=group_name,
        epochs=cli.epochs,
        parallel_actors=cli.parallel_actors,
        train_timesteps_per_epoch=cli.timesteps_per_epoch,
        train_grad_updates_per_epoch=cli.grads_per_epoch,
        start_learning_after_epoch=cli.start_learning_after_epoch,
        val_interval=cli.val_interval,
        ckpt_interval=cli.ckpt_interval,
        half_precision=cli.half_precision,
        fast_inference=not cli.slow_inference,
        batch_size=cli.batch_size,
        #memory_len=cli.memory_len,
        #agent_network=cli.agent_network,
        #weighted_sample=cli.weighted_sample,
        **extra_experiment_kwargs,
    )

    return experiment


