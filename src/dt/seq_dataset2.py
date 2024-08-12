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
    learning_history_dirs: str | List[str] = "saved_data/learning_history/ppo",
    max_episodes: int = -1,
    filter_episodes: int = 1,
    episode_lenght: int = 20,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Load learning history for the specified goal

    Args:
        goal_idx (int, optional): goal index. Defaults to 0.
        exp_name (str, optional): name of the experiment. Defaults to "darkroom".
        learning_history_dirs (str | List[str], optional): directory or list of directories where 
            to search for the files. Defaults to "saved_data/learning_history/ppo".
        max_episodes (int, optional): limit max number of episodes to load, if -1 all episoded 
            will be loaded. Defaults to -1 (no effect).
        filter_episodes (int, optional): filter out episodes to lower the number of episodes 
            (each `filter_episodes`-th will be taken). Defauts to 1 (no effect).
        episode_lenght (int, optional): lenght of the episode in the data. Defaults to 20.
        TODO save episode length in the folder with learning histories as parameter.

    Returns:
        Tuple[Dict[str, np.ndarray], Dict[str, Any]]: _description_
    """
    if isinstance(learning_history_dirs, str):
        learning_history_dirs = [learning_history_dirs]

    # Search for files with learning histories for the specified goals    
    filepaths = [os.path.join(learning_history_dir, x) for learning_history_dir in learning_history_dirs \
        for x in os.listdir(learning_history_dir) if f"{exp_name}-goal={goal_idx:02d}" in x]
    assert len(filepaths) == len(learning_history_dirs), f"The input directories ({learning_history_dirs}) "\
        f"contain more than 1 history or no histories ({filepaths}). Check the data in input directories."

    # Read and concatenate the files    
    cols = ["observations", "actions", "rewards"]
    data = [np.load(path) for path in filepaths]
    data = {col : np.concatenate([data[i][col] for i in range(len(filepaths))], axis=1)
            for col in cols}
    
    recorded_steps, num_actors = data['observations'].shape
    assert recorded_steps % episode_lenght == 0, f"Number of {recorded_steps} is not divisible by"\
        f"the {episode_lenght=}"
    
    def transform_traj(traj: np.ndarray) -> np.ndarray:
        """preprocess loaded trajectory"""
        traj = traj.reshape(-1, episode_lenght, num_actors)  # [episode, step, actor] 
        traj = traj[::filter_episodes]
        if max_episodes != -1:
            traj = traj[:max_episodes]
        traj = traj.transpose(2, 0, 1)  # [actor, episode, step]
        assert traj.shape[0] == num_actors
        traj = traj.reshape(num_actors, -1)  # [actor, step]
        return traj
    
    # Preprocess (reshape, filter, ...) the trajectories
    data = {traj_name : transform_traj(traj) for traj_name, traj in data.items()}
             
    info = {
        "num_actors": num_actors, 
        "num_episodes": data['rewards'].shape[1] // episode_lenght,
        "traj_lens": episode_lenght,
    }
    return data, info

class OneGoalSequenceDataset(IterableDataset):
    def __init__(self, 
            goal_idx: int = 0, 
            exp_name: str = "darkroom", 
            learning_history_dirs: str | List[str] = "saved_data/learning_history", 
            seq_len: int = 100, 
            actor: int | Literal['all'] = 'all',
            max_episodes: int = -1, 
            filter_episodes: int = 1,
            episode_length: int = 20,
        ) -> None:
        self.dataset, info = load_learning_history(
            goal_idx, exp_name, learning_history_dirs, max_episodes, filter_episodes, episode_length)
        
        self.seq_len = seq_len

        self.actor = actor
        self.num_actors = info["num_actors"]
        if isinstance(actor, int): assert actor < self.num_actors

    def _prepare_sample(self, begin_step_idx, actor_idx):
        end_step_idx = begin_step_idx + self.seq_len
        states = self.dataset["observations"][actor_idx, begin_step_idx:end_step_idx].flatten().astype(int)
        actions = self.dataset["actions"][actor_idx, begin_step_idx:end_step_idx].flatten().astype(int)
        rewards = self.dataset["rewards"][actor_idx, begin_step_idx:end_step_idx].flatten().astype(np.float32)
        return states, actions, rewards
    
    @property
    def max_step_idx(self):
        return self.dataset['observations'].shape[1] - self.seq_len
    
    def get_random_sample(self):
        begin_step_idx = random.randint(0, self.max_step_idx)
        actor_idx = self.actor if isinstance(self.actor, int) else random.randint(0, self.num_actors - 1)
        return self._prepare_sample(begin_step_idx, actor_idx)

    def __iter__(self):
        while True:
            yield self.get_random_sample()
            
class SequenceDataset(IterableDataset):
    def __init__(self, 
            goal_idxs: List[int] = [0], 
            exp_name: str = "darkroom", 
            learning_history_dirs: str | List[str] = "saved_data/learning_history/ppo", 
            seq_len: int = 100, 
            actor: int | Literal['all'] = 'all',
            max_episodes: int = -1, 
            filter_episodes: int = 1,
            episode_length: int = 20,
        ) -> None:
        
        self.goal_idxs = goal_idxs
        self.data = {goal_idx : 
            OneGoalSequenceDataset(
                goal_idx, exp_name, learning_history_dirs, seq_len, actor,
                max_episodes, filter_episodes, episode_length)
            for goal_idx in goal_idxs}

    def __iter__(self):
        while True:
            goal_idx = np.random.choice(self.goal_idxs)
            yield self.data[goal_idx].get_random_sample()      
