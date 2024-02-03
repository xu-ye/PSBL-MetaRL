import os
import time
import random
from uuid import uuid4

import gymnasium as gym
import numpy as np
import torch
import gin
from einops import rearrange

from PSBL.loading import MAGIC_PAD_VAL
from PSBL.hindsight import Timestep, Trajectory, GoalSeq


def space_convert(gym_space):
    import gym as og_gym

    if isinstance(gym_space, og_gym.spaces.Box):
        return gym.spaces.Box(
            shape=gym_space.shape, low=gym_space.low, high=gym_space.high
        )
    elif isinstance(gym_space, og_gym.spaces.Discrete):
        return gym.spaces.Discrete(gym_space.n)
    elif isinstance(gym_space, gym.spaces.Space):
        return gym_space
    
    else:
        return gym.spaces.Box(
            shape=gym_space.shape, low=gym_space.low, high=gym_space.high
        )
        raise TypeError(f"Unsupported original gym space `{type(gym_space)}`")


class DiscreteActionWrapper(gym.ActionWrapper):
    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def action(self, action):
        if isinstance(action, int):
            return action
        if len(action.shape) > 0:
            action = action[0]
        action = int(action)
        return action


class ContinuousActionWrapper(gym.ActionWrapper):
    """
    Normalize continuous action spaces [-1, 1]
    """

    def __init__(self, env):
        super().__init__(env)
        self._true_action_space = env.action_space
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=self._true_action_space.shape,
            dtype=np.float32,
        )

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def action(self, action):
        true_delta = self._true_action_space.high - self._true_action_space.low
        norm_delta = self.action_space.high - self.action_space.low
        action = (action - self.action_space.low) / norm_delta
        action = action * true_delta + self._true_action_space.low
        return action


class GPUSequenceBuffer:
    def __init__(self, device, max_len: int, num_parallel: int,episode_length:int,horizon:int):
        self.device = device
        self.max_len = max_len
        self.num_parallel = num_parallel
        self.buffers = [None for _ in range(num_parallel)]
        self.cur_idxs = [0 for _ in range(num_parallel)]
        self.time_start = [0 for _ in range(num_parallel)]
        self.time_end = [1 for _ in range(num_parallel)]
        self.time_end_t = [1 for _ in range(num_parallel)]
        self._time_idx_buffer = torch.zeros(
            (num_parallel, max_len), device=self.device, dtype=torch.long
        )
        self._time_idx_buffer_test = torch.zeros(
            (num_parallel, max_len*50), device=self.device, dtype=torch.long
        )
        self.episode_length=episode_length+1
        self.new_longest=0
        self.horizon=horizon

    def _make_blank_buffer(self, arr):
        shape = (self.max_len,) + arr.shape[1:]
        return torch.full(shape, MAGIC_PAD_VAL).to(dtype=arr.dtype, device=self.device)
    def _make_blank_buffer_test(self, arr):
        shape = (50*self.max_len,) + arr.shape[1:]
        return torch.full(shape, MAGIC_PAD_VAL).to(dtype=arr.dtype, device=self.device)

    def add_timestep(self, arrays, dones=None):
        assert arrays.shape[0] == self.num_parallel
        assert arrays.shape[1] == 1

        if dones is None:
            dones = [False for _ in range(self.num_parallel)]
        arrays = torch.from_numpy(arrays).to(self.device)

        for i in range(arrays.shape[0]):
            self.time_end[i] += 1
            if dones[i] or self.buffers[i] is None:
                self.buffers[i] = self._make_blank_buffer(arrays[i])
                self.cur_idxs[i] = 0
                self.time_start[i] = 0
                self.time_end[i] = 1
            if self.cur_idxs[i] < self.max_len:
                self.buffers[i][self.cur_idxs[i]] = arrays[i]
                self.cur_idxs[i] += 1
            else:
                self.time_start[i] += 1
                self.buffers[i] = torch.cat((self.buffers[i], arrays[i]), axis=0)[
                    -self.max_len :
                ]
            
    def add_timestep_test(self, arrays, dones=None):
        assert arrays.shape[0] == self.num_parallel
        assert arrays.shape[1] == 1

        if dones is None:
            dones = [False for _ in range(self.num_parallel)]
        arrays = torch.from_numpy(arrays).to(self.device)

        for i in range(arrays.shape[0]):
            self.time_end[i] += 1
            self.time_end_t[i]+=1
            if dones[i] or self.buffers[i] is None:
                self.buffers[i] = self._make_blank_buffer_test(arrays[i])
                self.cur_idxs[i] = 0
                self.time_start[i] = 0
                self.time_end[i] = 1
                self.time_end_t[i] = 2
            if self.cur_idxs[i] < 50*self.max_len:
                self.buffers[i][self.cur_idxs[i]] = arrays[i]
                self.cur_idxs[i] += 1
            else:
                #self.time_start[i] += 1
                self.time_end[i] -= 1
                self.buffers[i] = torch.cat((self.buffers[i], arrays[i]), axis=0)[
                    -self.max_len :
                ]
        
    def add_timestep_test2(self, arrays, dones=None):
        assert arrays.shape[0] == self.num_parallel
        assert arrays.shape[1] == 1

        if dones is None:
            dones = [False for _ in range(self.num_parallel)]
        arrays = torch.from_numpy(arrays).to(self.device)

        for i in range(arrays.shape[0]):
            self.time_end[i] += 1
            if dones[i] or self.buffers[i] is None:
                self.buffers[i] = self._make_blank_buffer(arrays[i])
                self.cur_idxs[i] = 0
                self.time_start[i] = 0
                self.time_end[i] = 1
            if self.cur_idxs[i] < 50*self.max_len:
                self.buffers[i][self.cur_idxs[i]] = arrays[i]
                self.cur_idxs[i] += 1
            else:
                self.time_start[i] += 1
                self.buffers[i] = torch.cat((self.buffers[i], arrays[i]), axis=0)[
                    -self.max_len :
                ]

    @property
    def sequences(self):
        longest = max(self.cur_idxs)
        return torch.stack(self.buffers, axis=0)[:, :longest]
    
    def sequences_test(self):
        #episode_length=15
        last_num=0
        longest = max(self.cur_idxs)
        self.new_longest=max(self.time_end_t)-1
        num=( self.new_longest-self.max_len-1)//self.episode_length+1
        #self.new_longest=longest2 # time_end

        if  self.new_longest>self.max_len+1:
            self.new_longest= self.new_longest-self.episode_length*num
            buffer=torch.stack(self.buffers, axis=0)
            #out=torch.cat((buffer[:,:episode_length],buffer[:,-(self.new_longest-episode_length):]),dim=1)
            out=torch.cat((buffer[:,:self.episode_length*last_num],buffer[:,longest-(self.new_longest-self.episode_length*last_num):longest]),dim=1)


            return out
            #return torch.stack(self.buffers, axis=0)[:, -self.new_longest :]
        else:
            return torch.stack(self.buffers, axis=0)[:, :self.new_longest]
    
    def sequences_test_rl2s(self):
        #episode_length=15
        last_num=0
        longest = max(self.cur_idxs)
        self.new_longest=max(self.time_end_t)-1
        num=( self.new_longest-self.max_len-1)//self.episode_length+1
        #self.new_longest=longest2 # time_end

        if  self.new_longest>self.max_len+1:
            self.new_longest= self.new_longest-self.episode_length*num
            buffer=torch.stack(self.buffers, axis=0)
            #out=torch.cat((buffer[:,:episode_length],buffer[:,-(self.new_longest-episode_length):]),dim=1)
            out=torch.cat((buffer[:,:self.episode_length*last_num],buffer[:,longest-(self.new_longest-self.episode_length*last_num):longest]),dim=1)
            
            if longest>self.horizon:
                num_time=( longest-self.horizon-1)//self.episode_length+2
                out[:,:,2]=out[:,:,2]-self.episode_length/(self.horizon)*num_time



            return out
            #return torch.stack(self.buffers, axis=0)[:, -self.new_longest :]
        else:
            return torch.stack(self.buffers, axis=0)[:, :self.new_longest]
    
    def sequences_test2(self):
        #episode_length=15
        last_num=5
        longest = max(self.cur_idxs)
        self.new_longest=max(self.time_end_t)-1
        num=( self.new_longest-self.max_len-1)//self.episode_length+1
        #self.new_longest=longest2 # time_end

        if  0:
            self.new_longest= self.new_longest-self.episode_length*num
            buffer=torch.stack(self.buffers, axis=0)
            #out=torch.cat((buffer[:,:episode_length],buffer[:,-(self.new_longest-episode_length):]),dim=1)
            out=torch.cat((buffer[:,:self.episode_length*last_num],buffer[:,longest-(self.new_longest-self.episode_length*last_num):longest]),dim=1)


            return out
            #return torch.stack(self.buffers, axis=0)[:, -self.new_longest :]
        else:
            return torch.stack(self.buffers, axis=0)[:, :self.new_longest]

    @property
    def time_idxs_test(self):
        longest=max(self.time_end_t)-1
        
        #longest = max(self.cur_idxs)
        if longest>self.max_len:
            longest=self.max_len
            longest=self.new_longest
            
            #time_intervals = zip(self.time_start, self.time_start+longest)
        
        time_intervals = zip(self.time_start, [self.new_longest]*len(self.time_start))
        for i, interval in enumerate(time_intervals):
            arange = torch.arange(*interval)
            self._time_idx_buffer_test[i, : len(arange)] = arange
        out = self._time_idx_buffer_test[:, :longest]
        return out
    @property
    def time_idxs_test_episode(self):
        longest=max(self.time_end_t)-1
        
        
        #longest = max(self.cur_idxs)
        if longest>self.max_len:
            longest=self.max_len
            longest=self.new_longest
            
            #time_intervals = zip(self.time_start, self.time_start+longest)
        num_episode=int((longest-1)//self.episode_length)
        time_intervals = zip(self.time_start, [self.episode_length]*len(self.time_start))
        for i, interval in enumerate(time_intervals):
            arange = torch.arange(*interval)
            for num_idex in range(num_episode):
                self._time_idx_buffer_test[i,num_idex*self.episode_length : (num_idex+1)*self.episode_length] = arange
                
            self._time_idx_buffer_test[i, num_episode*self.episode_length: longest] = arange[:longest-num_episode*self.episode_length]
        out = self._time_idx_buffer_test[:, :longest]
        return out
    @property
    def time_idxs_test2(self):
        longest=max(self.time_end_t)-1
        
        #longest = max(self.cur_idxs)
        if longest>self.max_len:
            longest=self.max_len
            #longest=self.new_longest
            
            #time_intervals = zip(self.time_start, self.time_start+longest)
        
        time_intervals = zip(self.time_start, [longest]*len(self.time_start))
        for i, interval in enumerate(time_intervals):
            arange = torch.arange(*interval)
            if self.new_longest>self.max_len:
                self._time_idx_buffer_test[i, : self.new_longest] = torch.cat((torch.zeros(self.new_longest-self.max_len),arange),dim=0)
            else:
                self._time_idx_buffer_test[i, : len(arange)] = arange
        out = self._time_idx_buffer_test[:, :self.new_longest]
        

        return out
    
    
    
    @property
    def time_idxs(self):
        
        longest = max(self.cur_idxs)
        time_intervals = zip(self.time_start, self.time_end)
        for i, interval in enumerate(time_intervals):
            arange = torch.arange(*interval)
            self._time_idx_buffer[i, : len(arange)] = arange
        out = self._time_idx_buffer[:, :longest]
        return out

    @property
    def sequence_lengths(self):
        # shaped for Batch, Length, Actions
        return torch.Tensor(self.cur_idxs).to(self.device).view(-1, 1, 1).long()
    
    def sequence_lengths_test(self):
        # shaped for Batch, Length, Actions
        length=torch.Tensor(self.cur_idxs).to(self.device).view(-1, 1, 1).long()
        #if length.shape
        return torch.Tensor(self.cur_idxs).to(self.device).view(-1, 1, 1).long()


@gin.configurable
class ExplorationWrapper(gym.ActionWrapper):
    def __init__(
        self,
        env: gym.Env,
        eps_start_start: float = 1.0,
        eps_start_end: float = 0.05,
        eps_end_start: float = 0.8,
        eps_end_end: float = 0.01,
        steps_anneal: int = 1_000_000,
    ):
        super().__init__(env)

        self.eps_start_start = eps_start_start
        self.eps_start_end = eps_start_end
        self.eps_end_start = eps_end_start
        self.eps_end_end = eps_end_end
        self.global_slope = (eps_start_start - eps_start_end) / steps_anneal
        self.discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        self.global_step = 0

    def reset(self, *args, **kwargs):
        out = super().reset(*args, **kwargs)
        self.global_multiplier = random.random()
        return out

    def current_eps(self, local_step: int, horizon: int):
        ep_start = max(
            self.eps_start_start - self.global_slope * self.global_step,
            self.eps_start_end,
        )
        ep_end = max(
            self.eps_start_end - self.global_slope * self.global_step, self.eps_end_end
        )
        local_progress = float(local_step) / horizon
        current = self.global_multiplier * (
            ep_start - ((ep_start - ep_end) * local_progress)
        )
        return current

    def action(self, a):
        noise = self.current_eps(self.env.step_count, self.env.horizon)
        if self.discrete:
            # epsilon greedy (DQN-style)
            num_actions = self.env.action_space.n
            random_action = random.randrange(0, num_actions)
            use_random = random.random() <= noise
            if use_random:
                expl_action = np.full_like(a, random_action)
            else:
                expl_action = a
            assert expl_action.dtype == np.uint8
        else:
            # random noise (TD3-style)
            expl_action = a + noise * np.random.randn(*a.shape)
            expl_action = np.clip(expl_action, -1.0, 1.0).astype(np.float32)
            assert expl_action.dtype == np.float32

        self.global_step += 1
        return expl_action

    @property
    def return_history(self):
        return self.env.return_history

    @property
    def success_history(self):
        return self.env.success_history


class ReturnHistory:
    def __init__(self, env_name):
        self.data = {}

    def add_score(self, env_name, score):
        if env_name in self.data:
            self.data[env_name].append(score)
        else:
            self.data[env_name] = [score]


SuccessHistory = ReturnHistory


class SequenceWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        save_every: tuple[int, int] | None = None,
        make_dset: bool = False,
        dset_root: str = None,
        dset_name: str = None,
        dset_split: str = None,
    ):
        super().__init__(env)

        self.make_dset = make_dset
        if make_dset:
            assert dset_root is not None
            assert dset_name is not None
            assert dset_split in ["train", "val", "test"]
            self.dset_write_dir = os.path.join(dset_root, dset_name, dset_split)
            if not os.path.exists(self.dset_write_dir):
                os.makedirs(self.dset_write_dir)
        else:
            self.dset_write_dir = None
        self.dset_root = dset_root
        self.dset_name = dset_name
        self.dset_split = dset_split
        self.save_every = save_every
        self.since_last_save = 0
        self._total_frames = 0
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            action_shape = self.env.action_space.n
        else:
            action_shape = self.env.action_space.shape[-1]
        rl2_shape = action_shape + 1 + 1 + 1
        self.gcrl2_space = gym.spaces.Dict(
            {
                "obs": self.env.observation_space,
                "goal": self.env.kgoal_space,
                "rl2": gym.spaces.Box(
                    shape=(rl2_shape,),
                    dtype=np.float32,
                    low=float("-inf"),
                    high=float("inf"),
                ),
            }
        )

    @property
    def step_count(self):
        return self.env.step_count

    @property
    def horizon(self):
        return self.env.horizon

    def reset_stats(self):
        # stores all of the success/return histories
        self.return_history = ReturnHistory(self.env_name)
        self.success_history = SuccessHistory(self.env_name)

    def reset(self, seed=None) -> Timestep:
        timestep = self.env.reset(seed=seed)
        self.active_traj = Trajectory(
            max_goals=self.env.max_goal_seq_length, timesteps=[timestep]
        )
        self.since_last_save = 0
        self.save_this_time = (
            random.randint(*self.save_every) if self.save_every else None
        )
        self.total_return = 0.0
        self._current_timestep = self.active_traj.make_sequence(last_only=True)
        return timestep.obs, {}

    def step(self, action):
        timestep, reward, terminated, truncated, info = self.env.step(action)
        self.total_return += reward
        self.active_traj.add_timestep(timestep)
        self.since_last_save += 1
        if timestep.terminal:
            self.return_history.add_score(self.env.env_name, self.total_return)
            success = (
                self.active_traj.is_success
                if "success" not in info
                else info["success"]
            )
            #self.total_return = 0.0
            self.success_history.add_score(self.env.env_name, success)
        save = (
            self.save_every is not None and self.since_last_save > self.save_this_time
        )
        if (timestep.terminal or save) and self.make_dset:
            #self.log_to_disk()
            self.log_to_disk_returns()
            self.active_traj = Trajectory(
                max_goals=self.env.max_goal_seq_length, timesteps=[timestep]
            )
        self._current_timestep = self.active_traj.make_sequence(last_only=True)
        self._total_frames += 1
        return timestep.obs, reward, terminated, truncated, info

    def log_to_disk(self):
        traj_name = f"{self.env.env_name.strip().replace('_', '')}_{uuid4().hex[:8]}_{time.time()}.traj"
        path = os.path.join(self.dset_write_dir, traj_name)
        self.active_traj.save_to_disk(path)
        self.since_last_save = 0
    
    def log_to_disk_returns(self):
        traj_name = f"{self.env.env_name.strip().replace('_', '')}_{uuid4().hex[:8]}_{self.total_return}_{time.time()}.traj"
        path = os.path.join(self.dset_write_dir, traj_name)
        self.active_traj.save_to_disk(path)
        self.since_last_save = 0

    def sequence(self):
        seq = self.active_traj.make_sequence()
        return seq

    @property
    def total_frames(self):
        return self._total_frames

    @property
    def current_timestep(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._current_timestep
