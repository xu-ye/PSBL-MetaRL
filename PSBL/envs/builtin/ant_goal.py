import random

import numpy as np

from PSBL.envs.builtin.ant import AntEnv


class AntGoalEnv(AntEnv):
    def __init__(self, max_episode_steps=200):
        self.set_task(self.sample_tasks(1)[0])
        self._max_episode_steps = max_episode_steps
        self.task_dim = 2
        self.H = max_episode_steps
        self.t =0
        super(AntGoalEnv, self).__init__()

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        xposafter = np.array(self.get_body_com("torso"))

        goal_reward = -np.sum(np.abs(xposafter[:2] - self.goal_pos))  # make it happy, not suicidal

        ctrl_cost = .1 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 0.0
        reward = goal_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        done = False
        ob = self._get_obs()
        self.t+=1
        truncated = self.t >= self.H
        
        return ob, reward, done,truncated, dict(
            goal_forward=goal_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            task=self.get_task()
        )

    def sample_tasks(self, num_tasks):
        a = np.array([random.random() for _ in range(num_tasks)]) * 2 * np.pi
        r = 3 * np.array([random.random() for _ in range(num_tasks)]) ** 0.5
        return np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)

    def set_task(self, task):
        self.goal_pos = task
    
    def reset_task(self, task):
        if task is None:
            task = self.sample_tasks(1)[0]
        self.set_task(task)
        # self.reset()

    def get_task(self):
        return np.array(self.goal_pos)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])
    def reset(self, new_task=True, **kwargs):
        if new_task:
            task=None
            self.reset_task(task)
            
        #self.has_key = False
        self.t = 0
        super().reset()
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.1, high=0.1, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs(), {}


class AntGoalOracleEnv(AntGoalEnv):
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            self.goal_pos,
        ])
