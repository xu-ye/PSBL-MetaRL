import matplotlib as mpl
import random
import numpy as np
import torch
#from utils import helpers as utl
import matplotlib.pyplot as plt
#import seaborn as sns

from gym import Env
from gym import spaces

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def semi_circle_goal_sampler():
    r = 1.0
    angle = random.uniform(0, np.pi)
    goal = r * np.array((np.cos(angle), np.sin(angle)))
    return goal


def circle_goal_sampler():
    r = 1.0
    angle = random.uniform(0, 2*np.pi)
    goal = r * np.array((np.cos(angle), np.sin(angle)))
    return goal


GOAL_SAMPLERS = {
    'semi-circle': semi_circle_goal_sampler,
    'circle': circle_goal_sampler,
}


class PointEnv(Env):
    """
    point robot on a 2-D plane with position control
    tasks (aka goals) are positions on the plane

     - tasks sampled from unit square
     - reward is L2 distance
    """

    def __init__(self, max_episode_steps=100, goal_sampler=None):
        if callable(goal_sampler):
            self.goal_sampler = goal_sampler
        elif isinstance(goal_sampler, str):
            self.goal_sampler = GOAL_SAMPLERS[goal_sampler]
        elif goal_sampler is None:
            self.goal_sampler = semi_circle_goal_sampler
        else:
            raise NotImplementedError(goal_sampler)

        self.reset_task()
        self.task_dim = 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
        # we convert the actions from [-1, 1] to [-0.1, 0.1] in the step() function
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        self._max_episode_steps = max_episode_steps
        self.H = max_episode_steps
        self.t=0

    def sample_task(self):
        goal = self.goal_sampler()
        return goal

    def set_task(self, task):
        self._goal = task

    def get_task(self):
        return self._goal

    def reset_task(self, task=None):
        if task is None:
            task = self.sample_task()
        self.set_task(task)
        return task

    def reset_model(self):
        self._state = np.zeros(2)
        return self._get_obs()

    def reset(self):
        return self.reset_model()

    def _get_obs(self):
        return np.copy(self._state)

    def step(self, action):

        action = np.clip(action, self.action_space.low, self.action_space.high)
        assert self.action_space.contains(action), action

        self._state = self._state + 0.1 * action
        reward = - np.linalg.norm(self._state - self._goal, ord=2)
        truncated = self.t >= self.H
        done=truncated
        self.t += 1
        done = False
        ob = self._get_obs()
        info = {'task': self.get_task()}
        return ob, reward, done,truncated, info

    


class SparsePointEnv(PointEnv):
    """ Reward is L2 distance given only within goal radius """

    def __init__(self, goal_radius=0.2, max_episode_steps=100, goal_sampler='semi-circle'):
        super().__init__(max_episode_steps=max_episode_steps, goal_sampler=goal_sampler)
        self.goal_radius = goal_radius
        self.reset_task()

    def sparsify_rewards(self, r):
        ''' zero out rewards when outside the goal radius '''
        mask = (r >= -self.goal_radius).astype(np.float32)
        r = r * mask
        return r

    def reset_model(self):
        self._state = np.array([0, 0])
        return self._get_obs()

    def step(self, action):
        ob, reward, done,truncated, d = super().step(action)
        sparse_reward = self.sparsify_rewards(reward)
        # make sparse rewards positive
        if reward >= -self.goal_radius:
            sparse_reward += 1
        d.update({'sparse_reward': sparse_reward})
        d.update({'dense_reward': reward})
        return ob, sparse_reward, done,truncated, d
    
    def reset(self, new_task=True, **kwargs):
        if new_task:
            task=None
            self.reset_task(task)
        self.t = 0
        super().reset()

        self.reset_model()


        return self._get_obs().copy(),{}
