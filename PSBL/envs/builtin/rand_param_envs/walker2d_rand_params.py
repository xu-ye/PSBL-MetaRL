import numpy as np

from PSBL.envs.builtin.rand_param_envs.base import RandomEnv
from PSBL.envs.builtin.rand_param_envs.gym import utils
try:
    from PSBL.envs.builtin.rand_param_envs import mujoco_py
    from PSBL.envs.builtin.rand_param_envs.mujoco_py.mjlib import mjlib
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(
            e))

class Walker2DRandParamsEnv(RandomEnv, utils.EzPickle):
    def __init__(self, log_scale_limit=3.0):
        self._max_episode_steps = 200
        self._elapsed_steps = -1  # the thing below takes one step
        self.t=0
        self.H = self._max_episode_steps
        RandomEnv.__init__(self, log_scale_limit, 'walker2d.xml', 5)
        
        utils.EzPickle.__init__(self)
        self.reset(new_task=True)
        
        

    def _step(self, a):
        posbefore = self.model.data.qpos[0, 0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.model.data.qpos[0:3, 0]
        alive_bonus = 1.0
        reward = ((posafter - posbefore) / self.dt)
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = not (height > 0.8 and height < 2.0 and ang > -1.0 and ang < 1.0)
        ob = self._get_obs()
        self._elapsed_steps += 1
        info = {'task': self.get_task()}
        if self._elapsed_steps == self._max_episode_steps:
            done = True
            info['bad_transition'] = True
        truncated = self.t >= self.H
        self.t += 1
        return ob, reward, done,truncated, info

    def _get_obs(self):
        qpos = self.model.data.qpos
        qvel = self.model.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        return self._get_obs()

    def reset(self, new_task=True):
        if new_task:
            task=None
            self.reset_task(task)
        self.t = 0    
        #ob = super()._reset()
        mjlib.mj_resetData(self.model.ptr, self.data.ptr)
        ob = self.reset_model()
        self._elapsed_steps = 0
        #ob=self.reset_model()
        return ob

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20
