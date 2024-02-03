from abc import ABC, abstractmethod
from typing import Callable
import math

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import gin

from PSBL.nets.goal_embedders import FFGoalEmb, TokenGoalEmb
from PSBL.nets.utils import InputNorm
from PSBL.nets import ff, cnn


@gin.configurable
class TstepEncoder(nn.Module, ABC):
    def __init__(self, obs_shape, goal_shape, rl2_shape, goal_emb_Cls=TokenGoalEmb):
        super().__init__()
        self.obs_shape = obs_shape
        self.goal_shape = goal_shape
        self.rl2_shape = rl2_shape
        self.goal_emb = goal_emb_Cls(goal_length=goal_shape[0], goal_dim=goal_shape[1])
        self.goal_emb_dim = self.goal_emb.goal_emb_dim

    def forward(self, obs, goals, rl2s):
        goal_rep = self.goal_emb(goals)
        B, L, *_ = obs.shape
        out = self.inner_forward(obs, goal_rep, rl2s)
        return out

    @abstractmethod
    def inner_forward(self, obs, goal_rep, rl2s):
        pass

    @property
    @abstractmethod
    def emb_dim(self):
        pass



@gin.configurable
class FFTstepEncoderBatch(TstepEncoder):
    def __init__(
        self,
        obs_shape,
        goal_shape,
        rl2_shape,
        n_layers: int = 2,
        d_hidden: int = 512,
        d_output: int = 256,
        norm: str = "layer",
        activation: str = "leaky_relu",
        hide_rl2s: bool = False,
    ):
        super().__init__(
            obs_shape=obs_shape, goal_shape=goal_shape, rl2_shape=rl2_shape
        )
        flat_obs_shape = math.prod(obs_shape)
        in_dim = flat_obs_shape  + self.rl2_shape[-1]
        self.in_norm = InputNorm(flat_obs_shape + self.rl2_shape[-1])
        self.base = ff.MLP(
            d_inp=in_dim,
            d_hidden=d_hidden,
            n_layers=n_layers,
            d_output=d_output,
            activation=activation,
        )
        self.out_norm = ff.Normalization(norm, d_output)
        self._emb_dim = d_output
        self.hide_rl2s = hide_rl2s

    def inner_forward(self, obs, goal_rep, rl2s):
        B, L, *_ = obs.shape
        if self.hide_rl2s:
            rl2s = rl2s * 0 #[reset reward time action ]
        flat_obs_rl2 = torch.cat((obs.view(B, L, -1).float(), rl2s[:,:,:3]), dim=-1)# 
        flat_obs_rl2 = self.in_norm(flat_obs_rl2)
        if self.training:
            self.in_norm.update_stats(flat_obs_rl2)
        obs_rl2_goals = torch.cat((flat_obs_rl2, goal_rep), dim=-1)
        out = self.out_norm(self.base(obs_rl2_goals))
        return out
    
    def AllEncoder(self, obs, rl2s,action):

        B, L, *_ = obs.shape
        if self.hide_rl2s:
            rl2s = rl2s * 0 #[reset reward time action ]
        flat_obs_rl2 = torch.cat((obs.view(B, L, -1).float(), rl2s[:,:,:3],action.view(B, L, -1)), dim=-1)# 
        flat_obs_rl2 = self.in_norm(flat_obs_rl2)
        if self.training:
            self.in_norm.update_stats(flat_obs_rl2)
        out = self.out_norm(self.base(flat_obs_rl2))
        return out
    
    def AllEncoder2(self, obs, rl2s,obs_t_1,rl2s_t_1):

        B, L, *_ = obs.shape
        if self.hide_rl2s:
            rl2s = rl2s * 0 #[reset reward time action ]
        flat_obs_rl2 = torch.cat((obs_t_1.view(B, L, -1).float(), rl2s_t_1[:,:,:3],rl2s[:,:,3:]), dim=-1)#
        flat_obs_rl2 = self.in_norm(flat_obs_rl2)
        if self.training:
            self.in_norm.update_stats(flat_obs_rl2)
        out = self.out_norm(self.base(flat_obs_rl2))
        return out

    @property
    def emb_dim(self):
        return self._emb_dim



