import itertools
import math
import random
from torch.nn import functional as F

import gym
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
from gym import spaces
import gym

#from PSBL.envs.helpers import helpers as utl

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GridNavi(gym.Env):
    def __init__(self, num_cells=5, max_episode_steps=15):
        super(GridNavi, self).__init__()

        #self.seed()
        self.num_cells = num_cells
        self.num_states = num_cells ** 2

        self._max_episode_steps = max_episode_steps
        self.step_count = 0

        self.observation_space = spaces.Box(low=0, high=self.num_cells - 1, shape=(2,))
        self.action_space = spaces.Discrete(5)  # noop, up, right, down, left
        self.task_dim = 2
        self.belief_dim = 25

        # possible starting states
        self.starting_state = (0.0, 0.0)

        # goals can be anywhere except on possible starting states and immediately around it
        self.possible_goals = list(itertools.product(range(num_cells), repeat=2))
        self.possible_goals.remove((0, 0))
        self.possible_goals.remove((0, 1))
        self.possible_goals.remove((1, 1))
        self.possible_goals.remove((1, 0))

        self.task_dim = 2
        self.num_tasks = self.num_states

        # reset the environment state
        self._env_state = np.array(self.starting_state)
        # reset the goal
        self._goal = self.reset_task()
        # reset the belief
        self._belief_state = self._reset_belief()
        self.H = max_episode_steps

    def reset_task(self, task=None):
        if task is None:
            self._goal = np.array(random.choice(self.possible_goals))
        else:
            self._goal = np.array(task)
        self._reset_belief()
        return self._goal

    def _reset_belief(self):
        self._belief_state = np.zeros((self.num_cells ** 2))
        for pg in self.possible_goals:
            idx = self.task_to_id(np.array(pg))
            self._belief_state[idx] = 1.0 / len(self.possible_goals)
        return self._belief_state

    def update_belief(self, state, action):

        on_goal = state[0] == self._goal[0] and state[1] == self._goal[1]

        # hint
        if action == 5 or on_goal:
            possible_goals = self.possible_goals.copy()
            possible_goals.remove(tuple(self._goal))
            wrong_hint = possible_goals[random.choice(range(len(possible_goals)))]
            self._belief_state *= 0
            self._belief_state[self.task_to_id(self._goal)] = 0.5
            self._belief_state[self.task_to_id(wrong_hint)] = 0.5
        else:
            self._belief_state[self.task_to_id(state)] = 0
            self._belief_state = np.ceil(self._belief_state)
            self._belief_state /= sum(self._belief_state)

        assert (1-sum(self._belief_state)) < 1e-4

        return self._belief_state

    def get_task(self):
        return self._goal.copy()

    def get_belief(self):
        return self._belief_state.copy()

    def reset(self, new_task=True, **kwargs):
        if new_task:
            task=None
            self.reset_task(task)
        self.t = 0
        #super().reset()
        self.step_count = 0
        self._env_state = np.array(self.starting_state)
        return self._env_state.copy(),{}

    def state_transition(self, action):
        """
        Moving the agent between states
        """

        if action == 1:  # up
            self._env_state[1] = min([self._env_state[1] + 1, self.num_cells - 1])
        elif action == 2:  # right
            self._env_state[0] = min([self._env_state[0] + 1, self.num_cells - 1])
        elif action == 3:  # down
            self._env_state[1] = max([self._env_state[1] - 1, 0])
        elif action == 4:  # left
            self._env_state[0] = max([self._env_state[0] - 1, 0])

        return self._env_state

    def step(self, action):

        if isinstance(action, np.ndarray) and action.ndim == 1:
            action = action[0]
        assert self.action_space.contains(action)

        done = False

        # perform state transition
        state = self.state_transition(action)

        # check if maximum step limit is reached
        self.step_count += 1
        if self.step_count >= self._max_episode_steps:
            done = True
        truncated = self.t >= self.H
        self.t += 1

        # compute reward
        if self._env_state[0] == self._goal[0] and self._env_state[1] == self._goal[1]:
            reward = 1.0
        else:
            reward = -0.1

        # update ground-truth belief
        self.update_belief(self._env_state, action)

        task = self.get_task()
        task_id = self.task_to_id(task)
        info = {'task': task,
                'task_id': task_id,
                'belief': self.get_belief()}
        return state, reward, done,truncated, info

    def task_to_id(self, goals):
        mat = torch.arange(0, self.num_cells ** 2).long().reshape((self.num_cells, self.num_cells))
        if isinstance(goals, list) or isinstance(goals, tuple):
            goals = np.array(goals)
        if isinstance(goals, np.ndarray):
            goals = torch.from_numpy(goals)
        goals = goals.long()

        if goals.dim() == 1:
            goals = goals.unsqueeze(0)

        goal_shape = goals.shape
        if len(goal_shape) > 2:
            goals = goals.reshape(-1, goals.shape[-1])

        classes = mat[goals[:, 0], goals[:, 1]]
        classes = classes.reshape(goal_shape[:-1])

        return classes

    def id_to_task(self, classes):
        mat = torch.arange(0, self.num_cells ** 2).long().reshape((self.num_cells, self.num_cells)).numpy()
        goals = np.zeros((len(classes), 2))
        classes = classes.numpy()
        for i in range(len(classes)):
            pos = np.where(classes[i] == mat)
            goals[i, 0] = float(pos[0][0])
            goals[i, 1] = float(pos[1][0])
        goals = torch.from_numpy(goals).to(device).float()
        return goals

    def goal_to_onehot_id(self, pos):
        cl = self.task_to_id(pos)
        if cl.dim() == 1:
            cl = cl.view(-1, 1)
        nb_digits = self.num_cells ** 2
        # One hot encoding buffer that you create out of the loop and just keep reusing
        y_onehot = torch.FloatTensor(pos.shape[0], nb_digits).to(device)
        # In your for loop
        y_onehot.zero_()
        y_onehot.scatter_(1, cl, 1)
        return y_onehot

    def onehot_id_to_goal(self, pos):
        if isinstance(pos, list):
            pos = [self.id_to_task(p.argmax(dim=1)) for p in pos]
        else:
            pos = self.id_to_task(pos.argmax(dim=1))
        return pos

    


def plot_rew_reconstruction(env,
                            rew_pred_means,
                            rew_pred_vars,
                            image_folder,
                            iter_idx,
                            ):
    """
    Note that env might need to be a wrapped env!
    """

    num_rollouts = len(rew_pred_means)

    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    test_rew_mus = torch.cat(rew_pred_means).cpu().detach().numpy()
    plt.plot(range(test_rew_mus.shape[0]), test_rew_mus, '.-', alpha=0.5)
    plt.plot(range(test_rew_mus.shape[0]), test_rew_mus.mean(axis=1), 'k.-')
    for tj in np.cumsum([0, *[env._max_episode_steps for _ in range(num_rollouts)]]):
        span = test_rew_mus.max() - test_rew_mus.min()
        plt.plot([tj + 0.5, tj + 0.5], [test_rew_mus.min() - 0.05 * span, test_rew_mus.max() + 0.05 * span], 'k--',
                 alpha=0.5)
    plt.title('output - mean')

    plt.subplot(1, 3, 2)
    test_rew_vars = torch.cat(rew_pred_vars).cpu().detach().numpy()
    plt.plot(range(test_rew_vars.shape[0]), test_rew_vars, '.-', alpha=0.5)
    plt.plot(range(test_rew_vars.shape[0]), test_rew_vars.mean(axis=1), 'k.-')
    for tj in np.cumsum([0, *[env._max_episode_steps for _ in range(num_rollouts)]]):
        span = test_rew_vars.max() - test_rew_vars.min()
        plt.plot([tj + 0.5, tj + 0.5], [test_rew_vars.min() - 0.05 * span, test_rew_vars.max() + 0.05 * span],
                 'k--', alpha=0.5)
    plt.title('output - variance')

    plt.subplot(1, 3, 3)
    rew_pred_entropy = -(test_rew_vars * np.log(test_rew_vars)).sum(axis=1)
    plt.plot(range(len(test_rew_vars)), rew_pred_entropy, 'r.-')
    for tj in np.cumsum([0, *[env._max_episode_steps for _ in range(num_rollouts)]]):
        span = rew_pred_entropy.max() - rew_pred_entropy.min()
        plt.plot([tj + 0.5, tj + 0.5], [rew_pred_entropy.min() - 0.05 * span, rew_pred_entropy.max() + 0.05 * span],
                 'k--', alpha=0.5)
    plt.title('Reward prediction entropy')

    plt.tight_layout()
    if image_folder is not None:
        plt.savefig('{}/{}_rew_decoder'.format(image_folder, iter_idx))
        plt.close()
    else:
        plt.show()


def plot_bb(env, args, episode_all_obs, episode_goals, reward_decoder,
            episode_latent_means, episode_latent_logvars, image_folder, iter_idx, episode_beliefs):
    """
    Plot behaviour and belief.
    """

    plt.figure(figsize=(1.5 * env._max_episode_steps, 1.5 * args.max_rollouts_per_task))

    num_episodes = len(episode_all_obs)
    num_steps = len(episode_all_obs[0])

    rew_pred_means = [[] for _ in range(num_episodes)]
    rew_pred_vars = [[] for _ in range(num_episodes)]

    # loop through the experiences
    for episode_idx in range(num_episodes):
        for step_idx in range(num_steps):

            curr_obs = episode_all_obs[episode_idx][:step_idx + 1]
            curr_goal = episode_goals[episode_idx]

            if episode_latent_means is not None:
                curr_means = episode_latent_means[episode_idx][:step_idx + 1]
                curr_logvars = episode_latent_logvars[episode_idx][:step_idx + 1]

            # choose correct subplot
            plt.subplot(args.max_rollouts_per_task,
                        math.ceil(env._max_episode_steps) + 1,
                        1 + episode_idx * (1 + math.ceil(env._max_episode_steps)) + step_idx),

            # plot the behaviour
            plot_behaviour(env, curr_obs, curr_goal)

            if reward_decoder is not None:
                # visualise belief in env
                rm, rv = compute_beliefs(env,
                                         args,
                                         reward_decoder,
                                         curr_means[-1],
                                         curr_logvars[-1],
                                         curr_goal)
                rew_pred_means[episode_idx].append(rm)
                rew_pred_vars[episode_idx].append(rv)
                plot_belief(env, rm, args)
            elif episode_beliefs is not None:
                curr_beliefs = episode_beliefs[episode_idx][step_idx]
                plot_belief(env, curr_beliefs, args)
            else:
                rew_pred_means = rew_pred_vars = None

            if episode_idx == 0:
                plt.title('t = {}'.format(step_idx))

            if step_idx == 0:
                plt.ylabel('Episode {}'.format(episode_idx + 1))

    if reward_decoder is not None:
        rew_pred_means = [torch.stack(r) for r in rew_pred_means]
        rew_pred_vars = [torch.stack(r) for r in rew_pred_vars]

    # save figure that shows policy behaviour
    plt.tight_layout()
    if image_folder is not None:
        plt.savefig('{}/{}_behaviour'.format(image_folder, iter_idx))
        plt.close()
    else:
        plt.show()

    return rew_pred_means, rew_pred_vars


def plot_behaviour(env, observations, goal):
    num_cells = int(env.observation_space.high[0] + 1)

    # draw grid
    for i in range(num_cells):
        for j in range(num_cells):
            pos_i = i
            pos_j = j
            rec = Rectangle((pos_i, pos_j), 1, 1, facecolor='none', alpha=0.5,
                            edgecolor='k')
            plt.gca().add_patch(rec)

    # shift obs and goal by half a stepsize
    if isinstance(observations, tuple) or isinstance(observations, list):
        observations = torch.cat(observations)
    observations = observations.cpu().numpy() + 0.5
    goal = np.array(goal) + 0.5

    # visualise behaviour, current position, goal
    plt.plot(observations[:, 0], observations[:, 1], 'b-')
    plt.plot(observations[-1, 0], observations[-1, 1], 'b.')
    plt.plot(goal[0], goal[1], 'kx')

    # make it look nice
    plt.xticks([])
    plt.yticks([])
    plt.xlim([0, num_cells])
    plt.ylim([0, num_cells])





def plot_belief(env, beliefs, args):
    """
    Plot the belief by taking 100 samples from the latent space and plotting the average predicted reward per cell.
    """

    num_cells = int(env.observation_space.high[0] + 1)
    unwrapped_env = env.venv.unwrapped.envs[0]

    # draw probabilities for each grid cell
    alphas = []
    for i in range(num_cells):
        for j in range(num_cells):
            pos_i = i
            pos_j = j
            idx = unwrapped_env.task_to_id(torch.tensor([[pos_i, pos_j]]))
            alpha = beliefs[idx]
            alphas.append(alpha.item())
    alphas = np.array(alphas)
    # cut off values (this only happens if we don't use sigmoid/softmax)
    alphas[alphas < 0] = 0
    alphas[alphas > 1] = 1
    # alphas = (np.array(alphas)-min(alphas)) / (max(alphas) - min(alphas))
    count = 0
    for i in range(num_cells):
        for j in range(num_cells):
            pos_i = i
            pos_j = j
            rec = Rectangle((pos_i, pos_j), 1, 1, facecolor='r', alpha=alphas[count],
                            edgecolor='k')
            plt.gca().add_patch(rec)
            count += 1
