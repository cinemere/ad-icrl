# %%
import os
import random
from typing import Any, Dict, List, Tuple, Literal

import numpy as np
from torch.utils.data import IterableDataset
# %%
def load_learning_history(
    goal_idx: int = 0,
    exp_name: str = "darkroom",
    learning_history_dir: str = "saved_data/learning_history/ppo",
    max_episodes: int = -1,
    filter_episodes: int = 1,
    episode_lenght: int = 20,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """_summary_

    Args:
        goal_idx (int, optional): _description_. Defaults to 0.
        exp_name (str, optional): _description_. Defaults to "darkroom".
        learning_history_dir (str, optional): _description_. Defaults to "saved_data/learning_history".
        max_episodes (int, optional): limit the number of loaded episodes. If -1, no effect.
        filter_episodes (int, optinal): filter out episodes to lower the number of 
            eposides (each `filter_episodes`-th will be taken). Defauts to 1 (no effect).
        
    Returns:
        Tuple[Dict[str, np.ndarray], Dict[str, Any]]: 
            - traj: trjectories dataset with shape [n_episodes, n_actors, episode_length]
            - info
    """
    # select filename, which holds the selected task trajectory
    filenames = [x for x in os.listdir(learning_history_dir) if f"{exp_name}-goal={goal_idx:02d}" in x]
    assert len(filenames) == 1
    filepath = os.path.join(learning_history_dir, filenames[0])


    data = np.load(filepath)
    recorded_steps, num_actors = data['observations'].shape
    max_num_episodes = recorded_steps // episode_lenght
    
    cols = ["observations", "actions", "rewards"]
    traj = {col : [] for col in cols}
    for col_name in cols:
        col = data[col_name][:max_num_episodes*episode_lenght:filter_episodes]
        col = col.reshape(-1, episode_lenght, num_actors)
        col = col.transpose(0, 2, 1)
        traj[col_name] = col
    
    info = {
        # "obs_mean": traj["observations"].mean(),
        # "obs_std": traj["observations"].std() + 1e-6,
        "reward_mean": traj['rewards'][-1].mean(),
        "num_actors": traj["observations"].shape[-2], 
        "traj_lens": traj["observations"].shape[-1],
    }
    return traj, info

class OneGoalSequenceDataset(IterableDataset):
    def __init__(self, 
            goal_idx: int = 0, 
            exp_name: str = "darkroom", 
            learning_history_dir: str = "saved_data/learning_history", 
            seq_len: int = 100, 
            actor: int | Literal['all'] = 'all',
            max_episodes: int = -1, 
            filter_episodes: int = 1,
        ) -> None:
        self.dataset, info = load_learning_history(goal_idx, exp_name, learning_history_dir, max_episodes, filter_episodes)
        self.seq_len = seq_len

        self.actor = actor
        self.num_actors = info["num_actors"]
        if isinstance(actor, int): assert actor < self.num_actors

        # self.state_mean = info["obs_mean"]
        # self.state_std = info["obs_std"]
        self.reward_scale = 1 / info["reward_mean"]

        assert seq_len % info["traj_lens"] == 0
        self.episode_length = info["traj_lens"]
        self.num_sampled_episodes = seq_len // info["traj_lens"]

    def __prepare_sample(self, episode_idx, actor_idx):
        states = self.dataset["observations"][episode_idx:episode_idx+self.num_sampled_episodes, actor_idx].flatten().astype(int)
        actions = self.dataset["actions"][episode_idx:episode_idx+self.num_sampled_episodes, actor_idx].flatten().astype(int)
        rewards = self.dataset["rewards"][episode_idx:episode_idx+self.num_sampled_episodes, actor_idx].flatten().astype(np.float32)
        # time_steps = np.arange(self.seq_len) + episode_idx * self.episode_length

        # states = (states - self.state_mean) / self.state_std
        # rewards = rewards * self.reward_scale
        
        return states, actions, rewards #, time_steps
    
    @property
    def max_episode_idx(self):
        return self.dataset['observations'].shape[0] - self.num_sampled_episodes
    
    def get_random_sample(self):
        episode_idx = random.randint(0, self.max_episode_idx)
        actor_idx = self.actor if isinstance(self.actor, int) else random.randint(0, self.num_actors - 1)
        return self.__prepare_sample(episode_idx, actor_idx)

    def __iter__(self):
        while True:
            yield self.get_random_sample()
            
class SequenceDataset(IterableDataset):
    def __init__(self, 
            goal_idxs: List[int] = [0], 
            exp_name: str = "darkroom", 
            learning_history_dir: str = "saved_data/learning_history/ppo", 
            seq_len: int = 100, 
            actor: int | Literal['all'] = 'all',
            max_episodes: int = -1, 
            filter_episodes: int = 1,
        ) -> None:
        
        self.goal_idxs = goal_idxs
        self.data = {goal_idx : OneGoalSequenceDataset(
                goal_idx, exp_name, learning_history_dir, seq_len, actor,
                max_episodes, filter_episodes)
            for goal_idx in goal_idxs}

    def __iter__(self):
        while True:
            goal_idx = np.random.choice(self.goal_idxs)
            yield self.data[goal_idx].get_random_sample()      