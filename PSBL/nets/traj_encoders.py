from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch import nn
import gin

from PSBL.nets import ff, transformer, utils
from PSBL.nets.actor_critic import MLP,DiscreteDist,ContinuousDist,ContinuousNormalDist,NormalDist,NormalDist_sample


class TrajEncoder(nn.Module, ABC):
    def __init__(self, tstep_dim: int, max_seq_len: int, horizon: int):
        super().__init__()
        self.tstep_dim = tstep_dim
        self.max_seq_len = max_seq_len
        self.horzion = horizon

    def reset_hidden_state(self, hidden_state, dones):
        return hidden_state

    def init_hidden_state(self, batch_size: int, device: torch.device):
        return None

    @abstractmethod
    def forward(self, seq: torch.Tensor, time_idxs: torch.Tensor, hidden_state=None):
        pass

    @abstractmethod
    def emb_dim(self):
        pass
#from PSBL.nets.transformer import TransformerOnlineAll




@gin.configurable
class TformerOnlineNewEncoderAll(TrajEncoder):
    def __init__(
        self,
        tstep_dim: int,
        max_seq_len: int,
        horizon: int,
        act_dim:int,
        obs_dim:int,
        discrete: bool,
        gammas: torch.Tensor,
        latent_dim:int=16,
        d_model: int = 256,
        n_heads: int = 8,
        d_ff: int = 1024,
        n_layers: int = 3,
        dropout_ff: float = 0.05,
        dropout_emb: float = 0.05,
        dropout_attn: float = 0.00,
        dropout_qkv: float = 0.00,
        activation: str = "leaky_relu",
        norm: str = "layer",
        attention: str = "flash",
        memory_len:int=150,
        episode_length:int=15,
        epsilon:float = 1e-6,
        
                ):
        super().__init__(tstep_dim, max_seq_len, horizon)
        self.tformer = transformer.TransformerOnlineAll(
            inp_dim=tstep_dim,
            max_pos_idx=horizon,
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            layers=n_layers,
            dropout_emb=dropout_emb,
            dropout_ff=dropout_ff,
            dropout_attn=dropout_attn,
            dropout_qkv=dropout_qkv,
            activation=activation,
            attention=attention,
            norm=norm,
            obs_dim=obs_dim,
            act_dim=act_dim,
            discrete=discrete,
            gammas=gammas,
            latent_dim=latent_dim,
            memory_len=memory_len,
            episode_length=episode_length,
        )
        self.d_model = d_model
        self.discrete=discrete
        
        self.action_scale = torch.tensor(1.)
        self.action_bias = torch.tensor(0.)
        self.epsilon=epsilon
        
        

    def init_hidden_state(self, batch_size: int, device: torch.device):
        def make_cache():
            return transformer.Cache(
                device=device,
                dtype=torch.bfloat16,
                batch_size=batch_size,
                max_seq_len=self.max_seq_len,
                n_heads=self.tformer.n_heads,
                head_dim=self.tformer.head_dim,
            )

        hidden_state = transformer.TformerHiddenState(
            key_cache=[make_cache() for _ in range(self.tformer.n_layers)],
            val_cache=[make_cache() for _ in range(self.tformer.n_layers)],
            timesteps=torch.zeros((batch_size,), dtype=torch.int32, device=device),
        )
        return hidden_state

    def reset_hidden_state(self, hidden_state, dones):
        if hidden_state is None:
            return None
        assert isinstance(hidden_state, transformer.TformerHiddenState)
        hidden_state.reset(idxs=dones)
        return hidden_state

    def forward(self, allseq, time_idxs, hidden_state=None):
        assert time_idxs is not None
        traj_emb, hidden_state,latent_dist_params,act_dist_params,r_dist_params,obs_dist_params=self.tformer(allseq, pos_idxs=time_idxs, hidden_state=hidden_state)
        a_z_r_o_output=(act_dist_params,latent_dist_params,r_dist_params,obs_dist_params)
        
        return traj_emb,hidden_state,a_z_r_o_output
    
    def get_action(self, allseq, time_idxs, hidden_state=None,sample: bool = True):
        assert time_idxs is not None
        traj_emb, hidden_state,latent_dist_params,act_dist_params,r_dist_params,obs_dist_params=self.tformer(allseq, pos_idxs=time_idxs, hidden_state=hidden_state)
        action_dist_para=act_dist_params[:,-1]
        if self.discrete:
            action_dist=DiscreteDist(action_dist_para)
        else:
            action_dist=ContinuousDist(action_dist_para)
        return latent_dist_params, hidden_state,action_dist
    
    def get_action_normal(self, allseq, time_idxs, hidden_state=None,sample: bool = True):
        assert time_idxs is not None
        traj_emb, hidden_state,latent_dist_params,act_dist_params,r_dist_params,obs_dist_params=self.tformer(allseq, pos_idxs=time_idxs, hidden_state=hidden_state)
        action_dist_para=act_dist_params[:,-1]
        if self.discrete:
            action_dist=DiscreteDist(action_dist_para)
        else:
            action_dist=NormalDist(action_dist_para)
        return latent_dist_params, hidden_state,action_dist
    
    def get_action_all(self, allseq, time_idxs, hidden_state=None,sample: bool = True):
        assert time_idxs is not None
        traj_emb, hidden_state,latent_dist_params,act_dist_params,r_dist_params,obs_dist_params=self.tformer.forward_all(allseq, pos_idxs=time_idxs, hidden_state=hidden_state)
        action_dist_para=act_dist_params[:,-1]
        if self.discrete:
            action_dist=DiscreteDist(action_dist_para)
        else:
            action_dist=ContinuousDist(action_dist_para)
        return latent_dist_params, hidden_state,action_dist



 

    
        


    @property
    def emb_dim(self):
        return self.d_model
    


   