from itertools import chain
from dataclasses import dataclass

import torch
from torch import nn
from einops import repeat, rearrange
import numpy as np
import gin

from PSBL.loading import Batch, MAGIC_PAD_VAL
from PSBL.nets.tstep_encoders import *
from PSBL.nets.traj_encoders import *
from PSBL.nets import actor_critic
import time
from PSBL.nets.actor_critic import NormalDist,NormalDist_sample

@gin.configurable
class Multigammas:
    def __init__(
        self,
        # fmt: off
        discrete = [0.7, 0.9, 0.93, 0.95, 0.98, 0.99, 0.992, 0.994, 0.995, 0.997, 0.998, 0.999, 0.9991, 0.9992, 0.9993, 0.9994, 0.9995],
        continuous = [0.9, 0.95, 0.99, 0.993, 0.996],
        discrete2=[.1, .9, .999],
        discrete3=[.1, .9, .95, .97, .99, .995],
        discrete4=[.1, .999],
        discrete5=[.1,0.9],
        discrete6=[.99,],
        discrete7=[0.7,.9,],

        # potential better default (work in progress):
        better2 = [.1, .9, .95, .97, .99, .995]
        # fmt: on
    ):
        #self.discrete = discrete2
        self.continuous = continuous
        self.discrete = discrete



@gin.configurable
class AgentOnlineAll(nn.Module):
    def __init__(
        self,
        obs_shape: tuple[int],
        goal_shape: tuple[int],
        rl2_shape: tuple[int],
        action_dim: int,
        max_seq_len: int,
        horizon: int,
        discrete: bool,
        alpha:float=0.2,
        tstep_encoder_Cls=FFTstepEncoderBatch,
        traj_encoder_Cls=TformerOnlineNewEncoderAll,
        num_critics: int = 4,
        num_critics_td: int = 2,
        online_coeff: float = 1.0,
        offline_coeff: float = 0.1,
        gamma: float = 0.999,
        reward_multiplier: float = 100.0,
        tau: float = 0.003,
        fake_filter: bool = False,
        popart: bool = True,
        use_target_actor: bool = True,
        use_multigamma: bool = True,
        latent_dim:int=256,
        init_learning_rate: float = 1e-4,
        gpu: int=0,
        automatic_entropy_tuning:bool=False,
        episode_length:int=50,
        meta_learning:bool=False,
        memory_len:int=150,
        
        episode_emb:bool=0,
        
    ):
        super().__init__()
        self.discrete = discrete
        self.obs_shape = obs_shape
        self.goal_shape = goal_shape
        self.rl2_shape = rl2_shape
        self.action_dim = action_dim
        self.reward_multiplier = reward_multiplier
        self.pad_val = MAGIC_PAD_VAL
        self.fake_filter = fake_filter
        self.offline_coeff = offline_coeff
        self.online_coeff = online_coeff
        self.tau = tau
        self.use_target_actor = use_target_actor
        assert num_critics_td <= num_critics
        self.num_critics_td = num_critics_td
        self.obs_dim=obs_shape[0]
        self.d_model=latent_dim
        self.alpha=alpha
        self.automatic_entropy_tuning=automatic_entropy_tuning
        self.gpu=gpu
        self.action_scale=1
        self.action_bias=0
        self.epsilon=1e-6
        self.episode_length=episode_length
        self.horizon=horizon
        self.meta_learning=meta_learning

        
        



        if discrete:
            multigammas = Multigammas().discrete
        else:
            multigammas = Multigammas().continuous
        # provided hparam `gamma` will stay in the -1 index
        # of gammas, actor, and critic outputs.
        gammas = (multigammas if use_multigamma else []) + [gamma]
        self.gammas = torch.Tensor(gammas).float()
        self.DEVICE = torch.device(f"cuda:{self.gpu}")


        self.tstep_encoder = FFTstepEncoderBatch(
            obs_shape=obs_shape, goal_shape=goal_shape, rl2_shape=rl2_shape
        )
        '''
        self.traj_encoder = TformerTrajEncoder(
            tstep_dim=self.tstep_encoder.emb_dim,
            max_seq_len=max_seq_len,
            horizon=horizon,
            
        )
        '''
        self.traj_encoder = TformerOnlineNewEncoderAll(
            tstep_dim=self.tstep_encoder.emb_dim,
            max_seq_len=max_seq_len,
            horizon=horizon,
            obs_dim=self.obs_dim,
            act_dim=action_dim,
            discrete=discrete,
            gammas=self.gammas,
            d_model=self.d_model,
            latent_dim=latent_dim,
        )
        self.latent_dim=latent_dim
        self.emb_dim = self.traj_encoder.emb_dim
        self.obs_dim=obs_shape[-1]
        self.target_traj_encoder = TformerOnlineNewEncoderAll(
            tstep_dim=self.tstep_encoder.emb_dim,
            max_seq_len=max_seq_len,
            horizon=horizon,
            obs_dim=self.obs_dim,
            act_dim=action_dim,
            discrete=discrete,
            gammas=self.gammas,
            d_model=self.d_model,
            latent_dim=latent_dim,
        )
        self.popart = actor_critic.PopArtLayer(gammas=len(gammas), enabled=popart)
        self.critics = actor_critic.NCritics(
            state_dim=self.latent_dim,
            action_dim=action_dim,
            discrete=discrete,
            num_critics=num_critics,
            gammas=self.gammas,
        )
        self.target_critics = actor_critic.NCritics(
            state_dim=self.latent_dim,
            action_dim=action_dim,
            discrete=discrete,
            num_critics=num_critics,
            gammas=self.gammas,
        )
        self.actor = actor_critic.Actor(
            state_dim=self.latent_dim,
            action_dim=action_dim,
            discrete=discrete,
            gammas=self.gammas,
        )
        self.sl_popart_obs = actor_critic.PopArtLayer(gammas=1, enabled=popart)
        self.sl_popart_r = actor_critic.PopArtLayer(gammas=1, enabled=popart)

        
        
        # full weight copy to targets
        self.hard_sync_targets()
        self.gaussian_loss=nn.GaussianNLLLoss()
        self.mse_loss=nn.MSELoss()
        self.gaussian_loss_n=nn.GaussianNLLLoss(reduction="none")

        if self.automatic_entropy_tuning is True:
                if discrete:
                    self.target_entropy = -np.log((1.0 / action_dim)) * 0.7
                else:
                    self.target_entropy = -torch.prod(torch.Tensor((action_dim,)).to(self.DEVICE)).item()
                
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.DEVICE)
                self.alpha_optim = torch.optim.AdamW([self.log_alpha], lr=init_learning_rate)
                self.alpha = self.log_alpha.exp()
        

    def get_current_timestep(self, sequences: torch.Tensor, seq_lengths: torch.Tensor):
        while sequences.ndim > seq_lengths.ndim:
            seq_lengths = seq_lengths.unsqueeze(-1)
        timesteps = torch.take_along_dim(sequences, seq_lengths - 1, dim=1)
        return timesteps

    @property
    def trainable_params(self):
        """
        Returns iterable of all trainable parameters that should be passed to an optimzer. (Everything but the target networks).
        """
        return chain(
            self.tstep_encoder.parameters(),
            self.traj_encoder.parameters(),
            self.critics.parameters(),
            #self.actor.parameters(),
        )

    def hard_sync_targets(self):
        """
        Hard copy online actor/critics to target actor/critics
        """
        for target_param, param in zip(
            self.target_traj_encoder.parameters(), self.traj_encoder.parameters()
        ):
            target_param.data.copy_(param.data)
        for target_param, param in zip(
            self.target_critics.parameters(), self.critics.parameters()
        ):
            target_param.data.copy_(param.data)
        #for target_param, param in zip(
        #    self.target_actor.parameters(), self.actor.parameters()
        #):
        #    target_param.data.copy_(param.data)

    def _ema_copy(self, target, online):
        for target_param, param in zip(target.parameters(), online.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def soft_sync_targets(self):
        """
        EMA copy online actor/critics to target actor/critics (DDPG-style)
        """
        self._ema_copy(self.target_traj_encoder, self.traj_encoder)
        self._ema_copy(self.target_critics, self.critics)
        #self._ema_copy(self.target_actor, self.actor)

    def get_actions(
        self,
        obs,
        goals,
        rl2s,
        seq_lengths,
        time_idxs,
        hidden_state=None,
        sample: bool = True,
    ):
        """
        Get rollout actions from the current policy.
        """
        using_hidden = hidden_state is not None
        if time_idxs[0,-1]==0:
            actions=torch.zeros_like(rl2s[:,:,3:]).to(obs.device).squeeze(1)
        else:
            obs=obs[:,:-1,:].clone()# 上一帧的obs s1 s2 s3
            goals=goals[:,:-1,:].clone()
            actions=rl2s[:,1:,3:].clone()
            rl2s=rl2s[:,:-1,:] #a0 a1 a2  r0 r1 r2

            #obs=obs[:,1:,:].clone()# 上一帧的obs s1 s2 s3
            #goals=goals[:,1:,:].clone()
            #actions=rl2s[:,1:,3:].clone()
            #rl2s=rl2s[:,1:,:] #a0 a1 a2  r0 r1 r2
            time_idxs=time_idxs[:,1:].clone()
            if using_hidden:
                obs = self.get_current_timestep(obs, seq_lengths-1)
                goals = self.get_current_timestep(goals, seq_lengths-1)
                rl2s = self.get_current_timestep(rl2s, seq_lengths-1)
                actions = self.get_current_timestep(actions, seq_lengths-1)
                time_idxs = self.get_current_timestep(time_idxs, seq_lengths.squeeze(-1)-1)
        
            all_seq=self.tstep_encoder.AllEncoder(obs=obs,rl2s=rl2s,action=actions)
            #all_seq = self.tstep_encoder(obs=obs, goals=goals, rl2s=rl2s)
            #act_emb=self.tstep_encoder.ActionEncoder(obs=obs,rl2s=rl2s)  ## 关注act 是否错位  关注第1步的时候
            #reward_obs_emb=self.tstep_encoder.RewardObsEncoder(obs=obs,rl2s=rl2s)

            # sequence model embedding [batch, length, d_emb]
            latent_dist_params, hidden_state,action_dists = self.traj_encoder.get_action(
                all_seq,time_idxs=time_idxs, hidden_state=hidden_state,sample=sample)
            #latent_dist_params, hidden_state = self.traj_encoder(
            #all_seq, time_idxs=time_idxs, hidden_state=hidden_state)
            #if not using_hidden:
            #    latent_emb_t = self.get_current_timestep(latent_dist_params, seq_lengths-1)
            #else:
            #latent_emb_t = latent_dist_params


            # generate action distribution [batch, len(self.gammas), d_action]
            #action_dists1 = self.actor(latent_emb_t.squeeze(1))

            if sample:
                actions = action_dists.sample()
            else:
                if self.discrete:
                    actions = torch.argmax(action_dists.probs, dim=-1, keepdim=True)
                else:
                    actions = action_dists.mean

            # get intended gamma distribution (always in -1 idx)
            actions = actions[..., -1, :]
        
        return actions, hidden_state

    def _td_stats(self, mask, q_s_a_g, r, td_target) -> dict:
        # messy data gathering for wandb console
        def masked_avg(x_, dim=0):
            return (mask[..., dim, :] * x_[..., dim, :]).sum().detach() / mask[
                ..., dim, :
            ].sum()

        q_seq = self.popart(q_s_a_g.detach(), normalized=False)
        stats = {}
        for i, gamma in enumerate(self.gammas):
            stats[f"q_s_a_g gamma={gamma}"] = masked_avg(q_s_a_g, i)
            stats[f"q_s_a_g (rescaled) gamma={gamma}"] = masked_avg(
                q_seq.mean(2, keepdims=True), i
            )
            stats[f"q_seq_mean gamma={gamma}"] = q_seq[..., i, :].mean(2)
            stats[f"q_seq_std gamma={gamma}"] = q_seq[..., i, :].std(2)

        stats.update(
            {
                "q_s_a_g unmasked std": q_s_a_g.std(),
                "min_td_target": (mask * td_target).min(),
                "mean_r": masked_avg(r),
                "td_target (target gamma)": masked_avg(td_target, -1),
                "real_return": torch.flip(
                    torch.cumsum(torch.flip(mask.all(2, keepdims=True) * r, (1,)), 1),
                    (1,),
                ).squeeze(-1),
                "q_s_a_g popart (target gamma)": masked_avg(self.popart(q_s_a_g), -1),
            }
        )
        return stats

    def _policy_stats(self, mask, a_dist) -> dict:
        # messy data gathering for wandb console
        # mask shape is batch length gammas 1
        sum_ = mask.sum((0, 1))
        masked_avg = (
            lambda x_, dim: (mask[..., dim, :] * x_[..., dim, :]).sum().detach() / sum_
        )

        if self.discrete:
            entropy = a_dist.entropy().unsqueeze(-1)
            low_prob = torch.min(a_dist.probs, dim=-1, keepdims=True).values
            high_prob = torch.max(a_dist.probs, dim=-1, keepdims=True).values
            return {
                "pi_entropy (target gamma)": masked_avg(entropy, -1),
                "pi_low_prob (target gamma)": masked_avg(low_prob, -1),
                "pi_high_prob (target gamma)": masked_avg(high_prob, -1),
                "pi_overall_high": (mask * a_dist.probs).max(),
            }
        else:
            entropy = -a_dist.log_prob(a_dist.sample()).sum(-1, keepdims=True)
            return {"pi_entropy (target_gamma)": masked_avg(entropy, -1)}

    def _filter_stats(self, mask, logp_a, filter_) -> dict:
        # messy data gathering for wandb console
        return {
            "filter": (mask[:, :-1, :] * filter_).sum() / mask[:, :-1, :].sum(),
            "min_logp_a": logp_a.min(),
            "max_logp_a": logp_a.max(),
        }
    
    def _sl_stats(self,obs_log,rewards_log,scaled_obs,scaled_rewards)-> dict:
        return {
            "obs_log": obs_log.mean().detach(),
            "rewards_log": rewards_log.mean().detach(),
            "scaled_obs": scaled_obs.mean().detach(),
            "scaled_rewards": scaled_rewards.mean().detach(),
        }
    def _incontext_stats(self,in_context_loss,)-> dict:
        return {
            "in_context_loss": in_context_loss.detach(),
            
        }
    def _alpha_stats(self,alpha_loss,alpha,entropy)-> dict:
        return {
            "alpha_loss": alpha_loss.detach(),
            "alpha_new": alpha.detach(),
            "entropy":entropy.detach().mean(),
        }


    def _popart_stats(self) -> dict:
        # messy data gathering for wandb console
        return {
            "popart_mu (mean over gamma)": self.popart.mu.data.mean().item(),
            "popart_nu (mean over gamma)": self.popart.nu.data.mean().item(),
            "popart_w (mean over gamma)": self.popart.w.data.mean().item(),
            "popart_b (mean over gamma)": self.popart.b.data.mean().item(),
            "popart_sigma (mean over gamma)": self.popart.sigma.mean().item(),
        }




@gin.configurable
class AgentOnlineAllTstep(nn.Module):
    def __init__(
        self,
        obs_shape: tuple[int],
        goal_shape: tuple[int],
        rl2_shape: tuple[int],
        action_dim: int,
        max_seq_len: int,
        horizon: int,
        discrete: bool,
        alpha:float=0.2,
        tstep_encoder_Cls=FFTstepEncoderBatch,
        traj_encoder_Cls=TformerOnlineNewEncoderAll,
        num_critics: int = 4,
        num_critics_td: int = 2,
        online_coeff: float = 1.0,
        offline_coeff: float = 0.1,
        gamma: float = 0.999,
        reward_multiplier: float = 100.0,
        tau: float = 0.003,
        fake_filter: bool = False,
        popart: bool = True,
        use_target_actor: bool = False,
        use_multigamma: bool = True,
        latent_dim:int=256,
        init_learning_rate: float = 1e-4,
        gpu: int=0,
        automatic_entropy_tuning:bool=False,
        episode_length:int=50,
        meta_learning:bool=False,
        memory_len:int=150,
        episode_emb:bool=0,

        
    ):
        super().__init__()
        self.discrete = discrete
        self.obs_shape = obs_shape
        self.goal_shape = goal_shape
        self.rl2_shape = rl2_shape
        self.action_dim = action_dim
        self.reward_multiplier = reward_multiplier
        self.pad_val = MAGIC_PAD_VAL
        self.fake_filter = fake_filter
        self.offline_coeff = offline_coeff
        self.online_coeff = online_coeff
        self.tau = tau
        self.use_target_actor = use_target_actor
        assert num_critics_td <= num_critics
        self.num_critics_td = num_critics_td
        self.obs_dim=obs_shape[0]
        self.d_model=latent_dim
        self.alpha=alpha
        self.automatic_entropy_tuning=automatic_entropy_tuning
        self.gpu=gpu
        self.episode_emb=episode_emb

        self.memory_len=memory_len

        self.action_scale=1
        self.action_bias=0
        self.epsilon=1e-6
        self.episode_length=episode_length
        self.horizon=horizon
        self.meta_learning=meta_learning


        
        self.sl_coef=0.5
        
        self.q_sac=0
        self.normal=0
        self.all_action=0

        #self.sl_obs_coef=0.15 #wind

        self.alpha=0
        self.use_popart_sl=False
        self.use_gaussian=False
        self.use_logprob=True

        #self.sl_rewads_coef=0.8 #gridworld
        #self.sl_obs_coef=0.2 #
        #self.r_bias_coef=10

        self.kl_coef=0.01
        self.use_kl=0

        




        if discrete:
            multigammas = Multigammas().discrete
        else:
            multigammas = Multigammas().continuous
        # provided hparam `gamma` will stay in the -1 index
        # of gammas, actor, and critic outputs.
        gammas = (multigammas if use_multigamma else []) + [gamma]
        self.gammas = torch.Tensor(gammas).float()
        self.DEVICE = torch.device(f"cuda:{self.gpu}")


        self.tstep_encoder = FFTstepEncoderBatch(
            obs_shape=obs_shape, goal_shape=goal_shape, rl2_shape=rl2_shape
        )
        '''
        self.traj_encoder = TformerTrajEncoder(
            tstep_dim=self.tstep_encoder.emb_dim,
            max_seq_len=max_seq_len,
            horizon=horizon,
            
        )
        '''
        self.traj_encoder = TformerOnlineNewEncoderAll(
            tstep_dim=self.tstep_encoder.emb_dim,
            max_seq_len=max_seq_len,
            horizon=horizon,
            obs_dim=self.obs_dim,
            act_dim=action_dim,
            discrete=discrete,
            gammas=self.gammas,
            d_model=self.d_model,
            latent_dim=latent_dim,
            memory_len=self.memory_len,
            episode_length=self.episode_length,
        )
        self.latent_dim=latent_dim
        self.emb_dim = self.traj_encoder.emb_dim
        self.obs_dim=obs_shape[-1]
        self.target_traj_encoder = TformerOnlineNewEncoderAll(
            tstep_dim=self.tstep_encoder.emb_dim,
            max_seq_len=max_seq_len,
            horizon=horizon,
            obs_dim=self.obs_dim,
            act_dim=action_dim,
            discrete=discrete,
            gammas=self.gammas,
            d_model=self.d_model,
            latent_dim=latent_dim,
            memory_len=self.memory_len,
            episode_length=self.episode_length,
        )
        self.popart = actor_critic.PopArtLayer(gammas=len(gammas), enabled=popart)
        self.critics = actor_critic.NCritics(
            state_dim=self.latent_dim,
            action_dim=action_dim,
            discrete=discrete,
            num_critics=num_critics,
            gammas=self.gammas,
        )
        self.target_critics = actor_critic.NCritics(
            state_dim=self.latent_dim,
            action_dim=action_dim,
            discrete=discrete,
            num_critics=num_critics,
            gammas=self.gammas,
        )
        self.actor = actor_critic.Actor(
            state_dim=self.latent_dim,
            action_dim=action_dim,
            discrete=discrete,
            gammas=self.gammas,
        )
        self.sl_popart_obs = actor_critic.PopArtLayer(gammas=1, enabled=popart)
        self.sl_popart_r = actor_critic.PopArtLayer(gammas=1, enabled=popart)

        
        
        # full weight copy to targets
        self.hard_sync_targets()
        self.gaussian_loss=nn.GaussianNLLLoss()
        self.mse_loss=nn.MSELoss()
        self.gaussian_loss_n=nn.GaussianNLLLoss(reduction="none")

        if self.automatic_entropy_tuning is True:
                if discrete:
                    self.target_entropy = -np.log((1.0 / action_dim)) * 0.7
                else:
                    self.target_entropy = -torch.prod(torch.Tensor((action_dim,)).to(self.DEVICE)).item()
                
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.DEVICE)
                self.alpha_optim = torch.optim.AdamW([self.log_alpha], lr=init_learning_rate)
                self.alpha = self.log_alpha.exp()
        

    def get_current_timestep(self, sequences: torch.Tensor, seq_lengths: torch.Tensor):
        while sequences.ndim > seq_lengths.ndim:
            seq_lengths = seq_lengths.unsqueeze(-1)
        timesteps = torch.take_along_dim(sequences, seq_lengths - 1, dim=1)
        return timesteps

    @property
    def trainable_params(self):
        """
        Returns iterable of all trainable parameters that should be passed to an optimzer. (Everything but the target networks).
        """
        return chain(
            self.tstep_encoder.parameters(),
            self.traj_encoder.parameters(),
            self.critics.parameters(),
            #self.actor.parameters(),
        )

    def hard_sync_targets(self):
        """
        Hard copy online actor/critics to target actor/critics
        """
        for target_param, param in zip(
            self.target_traj_encoder.parameters(), self.traj_encoder.parameters()
        ):
            target_param.data.copy_(param.data)
        for target_param, param in zip(
            self.target_critics.parameters(), self.critics.parameters()
        ):
            target_param.data.copy_(param.data)
        #for target_param, param in zip(
        #    self.target_actor.parameters(), self.actor.parameters()
        #):
        #    target_param.data.copy_(param.data)

    def _ema_copy(self, target, online):
        for target_param, param in zip(target.parameters(), online.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def soft_sync_targets(self):
        """
        EMA copy online actor/critics to target actor/critics (DDPG-style)
        """
        self._ema_copy(self.target_traj_encoder, self.traj_encoder)
        self._ema_copy(self.target_critics, self.critics)
        #self._ema_copy(self.target_actor, self.actor)

    def get_actions(
        self,
        obs,
        goals,
        rl2s,
        seq_lengths,
        time_idxs,
        hidden_state=None,
        sample: bool = True,
    ):
        """
        Get rollout actions from the current policy.
        """
        using_hidden = hidden_state is not None
        if 0:
            actions=torch.zeros_like(rl2s[:,:,3:]).to(obs.device).squeeze(1)
        else:
            #obs=obs[:,:-1,:].clone()# 上一帧的obs s1 s2 s3
            #goals=goals[:,:-1,:].clone()
            #actions=rl2s[:,1:,3:].clone()
            #rl2s=rl2s[:,:-1,:] #a0 a1 a2  r0 r1 r2
            obs_t_1=torch.cat((obs[:,0:1,:].clone(),obs[:,:-1,:]),dim=1)
            rl2s_t_1=torch.cat((rl2s[:,0:1,:].clone(),rl2s[:,:-1,:]),dim=1)

            #obs=obs[:,1:,:].clone()# 上一帧的obs s1 s2 s3
            #goals=goals[:,1:,:].clone()
            #actions=rl2s[:,1:,3:].clone()
            #rl2s=rl2s[:,1:,:] #a0 a1 a2  r0 r1 r2
            #time_idxs=time_idxs[:,1:].clone()
            if using_hidden:
                obs = self.get_current_timestep(obs, seq_lengths)
                goals = self.get_current_timestep(goals, seq_lengths)
                rl2s = self.get_current_timestep(rl2s, seq_lengths)
                obs_t_1 = self.get_current_timestep(obs_t_1, seq_lengths)
                rl2s_t_1 = self.get_current_timestep(rl2s_t_1, seq_lengths)
                #actions = self.get_current_timestep(actions, seq_lengths)
                time_idxs = self.get_current_timestep(time_idxs, seq_lengths.squeeze(-1))
        
            all_seq=self.tstep_encoder.AllEncoder2(obs=obs,rl2s=rl2s,obs_t_1=obs_t_1,rl2s_t_1=rl2s_t_1)
            #all_seq = self.tstep_encoder(obs=obs, goals=goals, rl2s=rl2s)
            #act_emb=self.tstep_encoder.ActionEncoder(obs=obs,rl2s=rl2s)  ## 关注act 是否错位  关注第1步的时候
            #reward_obs_emb=self.tstep_encoder.RewardObsEncoder(obs=obs,rl2s=rl2s)

            # sequence model embedding [batch, length, d_emb]
            if self.normal:
                latent_dist_params, hidden_state,action_dists = self.traj_encoder.get_action_normal(
                    all_seq,time_idxs=time_idxs, hidden_state=hidden_state,sample=sample)
            else:
                if self.all_action:
                    latent_dist_params, hidden_state,action_dists = self.traj_encoder.get_action_all(
                    all_seq,time_idxs=time_idxs, hidden_state=hidden_state,sample=sample)
                else:
                    
                    latent_dist_params, hidden_state,action_dists = self.traj_encoder.get_action(
                    all_seq,time_idxs=time_idxs, hidden_state=hidden_state,sample=sample)
            
            #latent_dist_params, hidden_state = self.traj_encoder(
            #all_seq, time_idxs=time_idxs, hidden_state=hidden_state)
            #if not using_hidden:
            #    latent_emb_t = self.get_current_timestep(latent_dist_params, seq_lengths-1)
            #else:
            #latent_emb_t = latent_dist_params


            # generate action distribution [batch, len(self.gammas), d_action]
            #action_dists1 = self.actor(latent_emb_t.squeeze(1))

            if sample:
                #actions = action_dists.sample()
                if self.normal:
                    actions,_ = NormalDist_sample(action_dists)
                else:
                    actions = action_dists.sample()
                    
            else:
                if self.discrete:
                    actions = torch.argmax(action_dists.probs, dim=-1, keepdim=True)
                else:
                    actions = action_dists.mean

            # get intended gamma distribution (always in -1 idx)
            actions = actions[..., -1, :]
        
        return actions, hidden_state

    def _td_stats(self, mask, q_s_a_g, r, td_target) -> dict:
        # messy data gathering for wandb console
        def masked_avg(x_, dim=0):
            return (mask[..., dim, :] * x_[..., dim, :]).sum().detach() / mask[
                ..., dim, :
            ].sum()

        q_seq = self.popart(q_s_a_g.detach(), normalized=False)
        stats = {}
        for i, gamma in enumerate(self.gammas):
            stats[f"q_s_a_g gamma={gamma}"] = masked_avg(q_s_a_g, i)
            stats[f"q_s_a_g (rescaled) gamma={gamma}"] = masked_avg(
                q_seq.mean(2, keepdims=True), i
            )
            stats[f"q_seq_mean gamma={gamma}"] = q_seq[..., i, :].mean(2)
            stats[f"q_seq_std gamma={gamma}"] = q_seq[..., i, :].std(2)

        stats.update(
            {
                "q_s_a_g unmasked std": q_s_a_g.std(),
                "min_td_target": (mask * td_target).min(),
                "mean_r": masked_avg(r),
                "td_target (target gamma)": masked_avg(td_target, -1),
                "real_return": torch.flip(
                    torch.cumsum(torch.flip(mask.all(2, keepdims=True) * r, (1,)), 1),
                    (1,),
                ).squeeze(-1),
                "q_s_a_g popart (target gamma)": masked_avg(self.popart(q_s_a_g), -1),
            }
        )
        return stats

    def _policy_stats(self, mask, a_dist) -> dict:
        # messy data gathering for wandb console
        # mask shape is batch length gammas 1
        sum_ = mask.sum((0, 1))
        masked_avg = (
            lambda x_, dim: (mask[..., dim, :] * x_[..., dim, :]).sum().detach() / sum_
        )

        if self.discrete:
            entropy = a_dist.entropy().unsqueeze(-1)
            low_prob = torch.min(a_dist.probs, dim=-1, keepdims=True).values
            high_prob = torch.max(a_dist.probs, dim=-1, keepdims=True).values
            return {
                "pi_entropy (target gamma)": masked_avg(entropy, -1),
                "pi_low_prob (target gamma)": masked_avg(low_prob, -1),
                "pi_high_prob (target gamma)": masked_avg(high_prob, -1),
                "pi_overall_high": (mask * a_dist.probs).max(),
            }
        else:
            entropy = -a_dist.log_prob(a_dist.sample()).sum(-1, keepdims=True)
            return {"pi_entropy (target_gamma)": masked_avg(entropy, -1)}

    def _filter_stats(self, mask, logp_a, filter_) -> dict:
        # messy data gathering for wandb console
        return {
            "filter": (mask[:, :-1, :] * filter_).sum() / mask[:, :-1, :].sum(),
            "min_logp_a": logp_a.min(),
            "max_logp_a": logp_a.max(),
        }
    
    def _sl_stats(self,obs_log,rewards_log,scaled_obs,scaled_rewards)-> dict:
        return {
            "obs_log": obs_log.mean().detach(),
            "rewards_log": rewards_log.mean().detach(),
            "scaled_obs": scaled_obs.mean().detach(),
            "scaled_rewards": scaled_rewards.mean().detach(),
        }
    def _kl_stats(self,kl_loss)-> dict:
        return {
            "kl_loss": kl_loss.mean().detach(),
        }
    def _incontext_stats(self,in_context_loss,)-> dict:
        return {
            "in_context_loss": in_context_loss.detach(),
            
        }
    def _alpha_stats(self,alpha_loss,alpha,entropy)-> dict:
        return {
            "alpha_loss": alpha_loss.detach(),
            "alpha_new": alpha.detach(),
            "entropy":entropy.detach().mean(),
        }


    def _popart_stats(self) -> dict:
        # messy data gathering for wandb console
        return {
            "popart_mu (mean over gamma)": self.popart.mu.data.mean().item(),
            "popart_nu (mean over gamma)": self.popart.nu.data.mean().item(),
            "popart_w (mean over gamma)": self.popart.w.data.mean().item(),
            "popart_b (mean over gamma)": self.popart.b.data.mean().item(),
            "popart_sigma (mean over gamma)": self.popart.sigma.mean().item(),
        }


    def compute_kl_loss(self, latent_mean, latent_logvar):
            # -- KL divergence
            if 0:
                kl_divergences = (- 0.5 * (1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp()).sum(dim=-1))
            else:
                gauss_dim = latent_mean.shape[-1]
                # add the gaussian prior
                device=latent_mean.device
                all_means = torch.cat((torch.zeros(1, *latent_mean.shape[1:]).to(device), latent_mean))
                all_logvars = torch.cat((torch.zeros(1, *latent_logvar.shape[1:]).to(device), latent_logvar))
                # https://arxiv.org/pdf/1811.09975.pdf
                # KL(N(mu,E)||N(m,S)) = 0.5 * (log(|S|/|E|) - K + tr(S^-1 E) + (m-mu)^T S^-1 (m-mu)))
                mu = all_means[1:]
                m = all_means[:-1]
                logE = all_logvars[1:]
                logS = all_logvars[:-1]
                kl_divergences = 0.5 * (torch.sum(logS, dim=-1) - torch.sum(logE, dim=-1) - gauss_dim + torch.sum(
                    1 / torch.exp(logS) * torch.exp(logE), dim=-1) + ((m - mu) / torch.exp(logS) * (m - mu)).sum(dim=-1))

            
            kl_divergences=kl_divergences.mean()*0.1

            return kl_divergences
  