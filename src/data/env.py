import os
from datetime import datetime
from dataclasses import dataclass
from gymnasium.wrappers.time_limit import TimeLimit
from toymeta.dark_room import DarkRoom
from typing import List

from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import RecordEpisodeStatistics

from src.data.generate_goals import get_all_goals

@dataclass
class SetupDarkRoom:
    # ---- env geometry params ----   
    
    size: int = 9
    "size of the room"
    
    goal_index: int | None = None
    "if set, the env will always be runned with the selected goal"
    
    random_start: bool = False
    "if False, the agent is starting from the center of the room"
    
    terminate_on_goal: bool = False
    "terminate the execution on reaching the goal"
    
    max_episode_lenght: int | None = 20
    "the length of the episode (TimeLimit wrapper)"
    
    # ---- logging params ----
    
    experiment_name: str = "darkroom"
    "experiment name (perfix)"

    add_now2name: bool = True
    "add now (datetime) to the experiment name"

    enable_monitor_logs: bool = False
    "enable sb3 Monitor wrapper (logs of reward, episode length and time)"
    
    enable_episode_stats_wrapper: bool = True
    "add gymnasium.wrappers.RecordEpisodeStatistics to initialization"
    
    monitor_logs_dir: str = "saved_data/logs/"
    "directory for putting sb3 Monitor (logs of reward, episode length and time)"

    def __post_init__(self):
        assert isinstance(self.goal_index, (type(None), int))
        self._setup_experiment_name()
        self._setup_monitor_logs_dir()
        
    def _setup_experiment_name(self):
        self.now = datetime.now().strftime(f"%d-%b-%H-%M-%S")
    
        if isinstance(self.goal_index, int): 
            self.experiment_name = f"{self.experiment_name}-goal={self.goal_index:02d}"
    
        if self.add_now2name:
            self.experiment_name = f"{self.experiment_name}_{self.now}"
            
    def _setup_monitor_logs_dir(self):
        self.monitor_logs_dir = os.path.join(self.monitor_logs_dir, self.experiment_name)
        os.makedirs(self.monitor_logs_dir, exist_ok=True)
    
    @property
    def goal(self) -> None | List[int]:
        if self.goal_index is None:
            return None

        return get_all_goals(self.size)[self.goal_index]
        
    def init_env(self, seed: int = 0) -> DarkRoom:
        env = DarkRoom(
            size = self.size,
            terminate_on_goal=self.terminate_on_goal,
            random_start=self.random_start,
            goal=self.goal,
        )
        
        if isinstance(self.max_episode_lenght, int):
            env = TimeLimit(env, max_episode_steps=self.max_episode_lenght)
        
        if self.enable_monitor_logs:  
            env = Monitor(env, self.monitor_logs_dir)
            
        if self.enable_episode_stats_wrapper:
            env = RecordEpisodeStatistics(env)
        
        env.reset(seed=seed)
        return env
    
    def init_n_envs(self, num_envs: int = 5, seed: int = 0):
        def init(seed):
            def _init():
                return self.init_env(seed)
            return _init
        return [init(seed + rank) for rank in range(num_envs)]
    
    @classmethod
    def get_cls(cls):
        return cls
    
    
def init_env(
    seed: int = 0,
    size: int = 9,
    # goal: None | np.ndarray = None,
    goal_idx: None | int = None,
    terminate_on_goal: bool = False,
    random_start: bool = False,
    max_episode_lenght: None | int = 20,
) -> DarkRoom:

    if isinstance(goal_idx, int):
        goal = get_all_goals(size=size)[goal_idx]
    else:
        goal = None

    env = DarkRoom(
        size = size,
        terminate_on_goal=terminate_on_goal,
        random_start=random_start,
        goal=goal,
    )
    
    if isinstance(max_episode_lenght, int):
        env = TimeLimit(env, max_episode_steps=max_episode_lenght)
    
    env.reset(seed=seed)
    return env
