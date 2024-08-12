# %%
from src.dt.seq_dataset2 import SequenceDataset
FILTER=2
sd = SequenceDataset(filter_episodes=FILTER)
# %%
it_sd = iter(sd)
# %%
outs = []
for _ in range(10_000):
    obs, act, rew = next(it_sd)
    outputs = rew.reshape(20, -1).sum(axis=0)
    outs.append((outputs - outputs[0])[1:])
    
# %%
import numpy as np

colors = plt.colormaps.get_cmap('viridis').resampled(4).colors
for i, color in enumerate(colors):
    values = [x[i] for x in outs]
    ending = "s" if i==0 else ""
    plt.hist(values, label=f"+{i+1} step{ending}", alpha=0.3, color=color)
    plt.axvline(np.mean(values), color=color, label=f"{np.mean(values):.1f}")
plt.legend()
plt.title(f"filter={FILTER}")
# %%
outs
# %%
# %%
import matplotlib.pyplot as plt
for i in range(5):
    plt.plot(sorted(sd.data[0].dataset['rewards'][:, i, :].sum(axis=1)))
# %%
import matplotlib.pyplot as plt
import numpy as np
rews = []
# path="/home/cinemere/work/repo/ad-icrl/saved_data/learning_history/ppo/darkroom-goal=00_31-Jul-14-24-54.npz"
path="/home/cinemere/work/repo/ad-icrl/saved_data/learning_history/ppo/darkroom-goal=41_31-Jul-14-48-27.npz"
arr = np.load(path)
obs = arr['observations']
act = arr['actions']
rew = arr['rewards']

sum_rewards = rew.reshape(-1, 20, 10).sum(axis=1)
for i in range(sum_rewards.shape[-1]):
    plt.plot(sum_rewards[:, i])
plt.plot(sum_rewards.mean(axis=-1))
# %%

max_episode_reward(41)
# %%
sum_rewards
# %%
A = act.reshape(-1, 20, 10)[-1, :, 0]
O = obs.reshape(-1, 20, 10)[-1, :, 0]
R = rew.reshape(-1, 20, 10)[-1, :, 0]
# %%
for (a, o, r) in zip(A, O, R):
    print(f"{a} {o:2d} {r}", get_all_goals(9)[o])
# %%
from src.collect_data.generate_goals import get_all_goals
size = 9

center_pos = (size // 2, size // 2)
goal_pos = get_all_goals(size)[41]
print(center_pos, goal_pos)
# %%
20 - 4 - 4
# %%

# %%
from dataclasses import dataclass
from typing import *

import sys
sys.path.append("/home/cinemere/work/repo/ad-icrl/cloned/headless_ad/envs")

@dataclass
class Config:
    # wandb params
    project: str = "AD"
    group: str = "gridworld-ad"
    job_type: str = "debug"
    name: Optional[str] = None

    # seeds
    train_seed: int = 0
    eval_seed: int = 100

    # data settings
    env_name: str = "GridWorld"
    num_train_envs: int = 10_000
    num_in_context_episodes: Optional[int] = None
    learning_histories_path: Optional[str] = "trajectories"

    num_eval_envs: int = 50
    eval_every: int = 1000
    log_every: int = 100

    num_train_steps: int = 100_000

    # Model Params
    seq_len: int = 100
    layer_norm_bias: bool = True
    token_embed_dim: int = 128
    d_model: int = 512
    num_layers: int = 2
    num_heads: int = 8
    dropout: float = 0.0
    attention_dropout: float = 0.0

    # Training params
    batch_size: int = 64
    learning_rate: float = 3e-3
    beta1: float = 0.9
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.1
    clip_grad_norm: float = 5
    get_action_type: str = "mode"
    rand_select_in_ucb: bool = True
    label_smoothing: float = 0.1

    # New
    rotary_percentage: float = 1.0  # is default in the llama's configs at the bottom
    parallel_residual: bool = False
    shared_attention_norm: bool = False
    _norm_class: str = "FusedRMSNorm"
    _mlp_class: str = "LLaMAMLP"

    # Device
    device: str = "cuda"
    autocast_dtype: str = "bf16"

    # Where to save data for experiment visualizations
    logs_dir: str = "logs"

    # New
    action_seq_len: int = 3
    train_frac_acts: float = 0.4
    train_frac_goals: float = 0.85
    grid_size: int = 9
    num_episodes: int = 200
    q_learning_lr: float = 0.9933
    q_learning_discount: float = 0.6238
    check_cached: bool = False

    action_space_type: str = "train"

    def __post_init__(self):
        self.job_type = self.action_space_type

        if self.num_in_context_episodes is None:
            self.num_in_context_episodes = 2 * self.num_episodes

        self.eval_seed = 1000 + self.train_seed

# %%
import warnings
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numba import njit
from itertools import product


def get_action_sequences(num_actions: int, seq_len: int):
    seqs = list(product(np.arange(num_actions), repeat=seq_len))
    seqs = np.vstack(seqs)

    return seqs

# gym warnings are annoying
warnings.filterwarnings("ignore")


@njit()
def pos_to_state(pos: Tuple[int, int], size: int):
    return int(pos[0] * size + pos[1])


@njit()
def single_step(
    action: int,
    agent_pos: np.ndarray,
    action_to_direction: np.ndarray,
    size: int,
    goal_pos: np.ndarray,
    terminate_on_goal: bool,
) -> Tuple[np.ndarray, Tuple[int, float, bool, bool]]:
    """
    This function makes an atomic step in the environment. Returns a new agent position and
    a usual tuple from a gym environment.

    :param action: index of an atomic action.
    :param agent_pos: the current position of an agent.
    :param action_to_direction: a list of transitions corresponding to each atomic action.
    :param size: the size of the grid.
    :param goal_pos: the goal's coordinates.
    :param terminate_on_goal: whether the episode ends upon reaching the goal.
    """

    agent_pos = np.clip(agent_pos + action_to_direction[action], 0, size - 1)

    reward = 1.0 if np.array_equal(agent_pos, goal_pos) else 0.0
    terminated = True if reward and terminate_on_goal else False

    gym_output = pos_to_state(agent_pos, size), reward, terminated, False

    return agent_pos, gym_output


@njit()
def multi_step(
    action: int,
    action_sequences: np.ndarray,
    agent_pos: np.ndarray,
    action_to_direction: np.ndarray,
    size: int,
    goal_pos: np.ndarray,
    terminate_on_goal: bool,
) -> Tuple[np.ndarray, Tuple[int, float, bool, bool]]:
    """
    This function makes an sequential step in the environment. Returns a new agent position and
    a usual tuple from a gym environment.

    :param action: index of a sequential action.
    :param action_sequences: for each sequential action specifies the sequence of atomic actions' indices.
    :param agent_pos: the current position of an agent.
    :param action_to_direction: a list of transitions corresponding to each atomic action.
    :param size: the size of the grid.
    :param goal_pos: the goal's coordinates.
    :param terminate_on_goal: whether the episode ends upon reaching the goal.
    """

    # Choose a sequence of atomic actions
    action_seq = action_sequences[action]

    # Perf each atomic action one after another
    rewards = np.zeros(len(action_seq))
    terms = np.zeros(len(action_seq))
    for i, act in enumerate(action_seq):
        agent_pos, gym_output = single_step(
            act, agent_pos, action_to_direction, size, goal_pos, terminate_on_goal
        )
        obs, rew, term, _ = gym_output
        rewards[i] = rew
        terms[i] = term

    # The reward will equal to 1 if the sequence's trajectory has passed
    # through a goal cell
    reward = int(np.any(rewards == 1))
    # The episode is finished if the sequence's trajectory has passed
    # through a goal cell
    term = np.any(terms)

    gym_output = obs, reward, term, False

    return agent_pos, gym_output


class GridWorld(gym.Env):
    """
    This is a darkroom environment where an agent operates in a grid and must reach a goal cell.
    A single action is a sequence of atomic actions 'noop', 'up', 'down', 'left' and 'right'.

    :param available_actions: indices of action sequences that the environment will use.
    :param action_seq_len: the amount of atomic actions constituting the action sequence.
    :param size: the size of the grid.
    :param goal_pos: the goal position. If None, will be chosen randomly.
    :param render_mode: same as in openai gym.
    :param terminate_on_goal: whether the episode ends upon reaching the goal.
    """

    def __init__(
        self,
        available_actions: np.ndarray,
        action_seq_len: int = 1,
        size: int = 9,
        goal_pos: Optional[np.ndarray] = None,
        render_mode=None,
        terminate_on_goal: bool = False,
    ):
        self.action_seq_len = action_seq_len
        # 5 is amount of atomic actions
        self.action_sequences = get_action_sequences(5, self.action_seq_len)
        self.action_sequences = self.action_sequences[available_actions]

        self.size = size
        self.observation_space = spaces.Discrete(self.size**2)
        self.action_space = spaces.Discrete(len(available_actions))

        self.action_to_direction = np.array([[0, 0], [-1, 0], [0, 1], [1, 0], [0, -1]])

        # the agent will start here
        self.center_pos = (self.size // 2, self.size // 2)

        # set the goal cell
        if goal_pos is not None:
            self.goal_pos = np.asarray(goal_pos)
            assert self.goal_pos.ndim == 1
        else:
            self.goal_pos = self.generate_goal_pos()

        self.terminate_on_goal = terminate_on_goal
        self.render_mode = render_mode

    def generate_goal_pos(self):
        """
        Generates random coordinates for the goal.
        """
        return self.np_random.integers(0, self.size, size=2)

    def state_to_pos(self, state):
        """
        Converts an index of a cell into 2-component coordinates
        """
        return np.array(divmod(state, self.size))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.agent_pos = np.array(self.center_pos, dtype=np.float32)

        return pos_to_state(self.agent_pos, self.size), {}

    def _single_step(self, action):
        """
        An atomic step in the environment.

        :param action: index of atomic action.
        """
        self.agent_pos, gym_output = single_step(
            action,
            agent_pos=self.agent_pos,
            action_to_direction=self.action_to_direction,
            size=self.size,
            goal_pos=self.goal_pos,
            terminate_on_goal=self.terminate_on_goal,
        )

        return gym_output + ({},)

    def step(self, action):
        """
        A 'sequential' step in an environment.

        :param action: index of a sequential action.
        """
        self.agent_pos, gym_output = multi_step(
            action,
            action_sequences=self.action_sequences,
            agent_pos=self.agent_pos,
            action_to_direction=self.action_to_direction,
            size=self.size,
            goal_pos=self.goal_pos,
            terminate_on_goal=self.terminate_on_goal,
        )

        return gym_output + ({},)

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == "rgb_array":
            # Create a grid representing the dark room
            grid = np.full(
                (self.size, self.size, 3), fill_value=(255, 255, 255), dtype=np.uint8
            )
            grid[self.goal_pos[0], self.goal_pos[1]] = (255, 0, 0)
            grid[int(self.agent_pos[0]), int(self.agent_pos[1])] = (0, 255, 0)
            return grid

# %%

# %%
config = Config()
output = generate_dataset(config=config, goals=np.ndarray([[0,0]]), actions=[0, 1, 2, 3, 4])

# %%
import random
from collections import defaultdict

def q_learning(
    env,
    lr=0.01,
    discount=0.9,
    num_episodes=int(1e7),
    seed=None,
):
    rng = np.random.default_rng(seed)
    Q = rng.uniform(size=(env.size * env.size, env.action_space.n))
    state, _ = env.reset(seed=seed)

    trajectories = defaultdict(list)
    save_filenames = []
    all_returns = []
    epsilons = []
    current_return = 0
    # creating lists to contain total rewards and steps per episode
    eps = 1.0
    eps_diff = 1.0 / (0.9 * num_episodes)
    episode_i = 0
    term, trunc = False, False
    i = 0
    while True:
        i += 1

        if term or trunc:
            all_returns.append(current_return)
            epsilons.append(eps)

            episode_i += 1
            eps = max(0, eps - eps_diff)
            current_return = 0
            # Get trajectories with optimal actions
            state, _ = env.reset()

        if random.random() < eps:
            a = rng.choice(env.action_space.n)
        else:
            a = Q[state, :].argmax()

        next_state, r, term, trunc, _ = env.step(a)
        current_return += r

        if term:
            Q[next_state, :] = 0

        # Collect trajectories with exploratory actions
        trajectories["states"].append(state)
        trajectories["actions"].append(a)
        trajectories["rewards"].append(r)
        trajectories["terminateds"].append(term)
        trajectories["truncateds"].append(trunc)
        trajectories["qtables"].append(Q)

        # Update Q-Table with new knowledge
        Q[state, a] += lr * (r + discount * np.max(Q[next_state, :]) - Q[state, a])
        state = next_state

        if episode_i == num_episodes:
            break

    all_returns = np.array(all_returns)
    epsilons = np.array(epsilons)

    return Q, all_returns, epsilons
# %%
from src.collect_data.collect import SetupDarkRoom

env_cfg = SetupDarkRoom(goal_index=0)
# %%
env = env_cfg.init_env()
# %%
env.reset()
# %%
import numpy as np
outputs = q_learning(env, num_episodes=10_000)
# %%
outputs[1]
# %%
import matplotlib.pyplot as plt
plt.plot(outputs[1])
# %% --------------------------------------------------------------------------------------------
# Show, what do we have in our dataset:

from src.collect_data.generate_goals import max_episode_reward
import matplotlib.pyplot as plt
import numpy as np
import os

# dir_path="/home/cinemere/work/repo/ad-icrl/saved_data/learning_history/ppo-24-Jul/"
# dir_path="/home/cinemere/work/repo/ad-icrl/saved_data/learning_history/ppo-short/"
# dir_path="/home/cinemere/work/repo/ad-icrl/saved_data/learning_history/ppo-01/"
# dir_path="/home/cinemere/work/repo/ad-icrl/saved_data/learning_history/ppo-02/"
dir_path="/home/cinemere/work/repo/ad-icrl/saved_data/learning_history/ppo-03/"
episode_length = 20

files = sorted(os.listdir(dir_path))

rewards = []
for index, file in enumerate(files):
    arr = np.load(os.path.join(dir_path, file))
    assert f"goal={index:02}" in file
    rewards.append(arr['rewards'] / max_episode_reward(index))
    # break

rewards = np.stack(rewards, axis=0)
num_goals, num_steps, num_workers = rewards.shape
# %%
rewards = rewards.reshape(num_goals, -1, episode_length, num_workers)
episode_rewards = rewards.sum(axis=-2).transpose(0, 2, 1)  # [goals, actors, episode]
# %% Plot mean over all goals and actors
mean_ep_rew = episode_rewards.reshape(num_goals * num_workers, -1)

plt.errorbar(
    np.arange(mean_ep_rew.shape[1]) * num_workers * episode_length, 
    mean_ep_rew.mean(0),
    mean_ep_rew.std(0))
plt.grid()
learning_history_dir = os.path.dirname(dir_path).split(os.sep)[-1]
plt.title("Mean over all goals and actors\n"\
    f"for learning history dataset in {learning_history_dir}")

plt.xlabel("overall timesteps")
plt.ylabel("normalized episode reward")
plt.show()
# %% Plot mean over all actors for each goal
mean_ep_rew_goals = episode_rewards.mean(axis=1)
std_ep_rew_goals  = episode_rewards.std(axis=1)

for i in range(num_goals):
    plt.errorbar(
        np.arange(mean_ep_rew_goals.shape[-1]), 
        mean_ep_rew_goals[i], 
        std_ep_rew_goals[i]
    )
plt.grid()
learning_history_dir = os.path.dirname(dir_path).split(os.sep)[-1]
plt.title("Mean over all actors\n"\
    f"for learning history dataset in {learning_history_dir}")

plt.xlabel("overall timesteps")
plt.ylabel("normalized episode reward")
plt.show()

# %% Which goals are not trained to oracle?

from src.collect_data.generate_goals import get_all_goals
for goal_index in range(num_goals):
    mean_episode_reward = episode_rewards.mean(axis=1)[goal_index, -1]
    if mean_episode_reward > 0.99:
        continue
    goal = get_all_goals()[goal_index].tolist()
    print(f"{goal_index=:02d} {goal=} {mean_episode_reward=:.4f}")

# %% Explore trajectory of episode

goal_index = 0  # select goal index
size = 9

goal = get_all_goals(size=size)[goal_index].tolist()
oracle_episode_reward = max_episode_reward(goal_idx=goal_index, size=size)
print(f"Selected {goal_index=}\n{goal=}\n{oracle_episode_reward=}")

# %%
file = [x for x in files if f"goal={goal_index:02d}" in x][0]
arr = np.load(os.path.join(dir_path, file))

rewards = arr['rewards'].reshape(-1, episode_length, num_workers)  # episode, step, worker
actions = arr['actions'].reshape(-1, episode_length, num_workers)
observations = arr['observations'].reshape(-1, episode_length, num_workers)

print("mean latest reward:", rewards.sum(axis=1)[-1].mean())
# %%
print(actions[-2, :, :10])
# %%
from src.collect_data.generate_goals import get_all_goals
img = np.zeros((size, size))
for i, step in enumerate(obs):
    x, y = get_all_goals(size)[step]
    img[x][y] = 50 + i
# %%
plt.imshow(img)
# %%
max_episode_reward(4)
# %%
no_duplicates_actions = []
no_duplicates_rewards = []
for i in range(100):
    _actions = actions[:, :, i]
    cond = _actions[:-1] == _actions[1:]
    _actions = _actions[1:][~cond.all(axis=1), :]
    no_duplicates_actions.append(_actions)
    no_duplicates_rewards.append(rewards[1:][~cond.all(axis=1), :, i])
# %%
for i in range(100):
    print(len(no_duplicates_actions[i]), no_duplicates_rewards[i].sum(axis=1))
# %%
path="/home/cinemere/work/repo/ad-icrl/saved_data/learning_history/sb-A2C/darkroom_goal=00_11-Jul-23-29-51/learning-history_episode-20000.npz"
# %%
import numpy as np

file = np.load(path)
# %%
r = file['rewards']
# %%
r.shape
# %%
r
# %%
