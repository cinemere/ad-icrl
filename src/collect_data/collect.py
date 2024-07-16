import os
from datetime import datetime
from dataclasses import dataclass, asdict
import yaml
import tyro
import gymnasium as gym
from gymnasium.wrappers.time_limit import TimeLimit
from toymeta.dark_room import DarkRoom
from typing import Literal, List
import numpy as np

from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CallbackList, BaseCallback, EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from .generate_goals import get_all_goals

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

    enable_monitor_logs: bool = True
    "enable sb3 Monitor wrapper (logs of reward, episode length and time)"
    
    monitor_logs_dir: str = "saved_data/logs/"
    "directory for putting sb3 Monitor (logs of reward, episode length and time)"

    def __post_init__(self):
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
        if isinstance(self.goal_index, int):
            return get_all_goals(self.size)[self.goal_index]
        return None
        
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
        
        env.reset(seed=seed)
        return env
    
    def init_n_envs(self, num_envs: int = 5, seed: int = 0):
        def init(seed):
            def _init():
                return self.init_env(seed)
            return _init
        return [init(seed + rank) for rank in range(num_envs)]


class LearningHistoryDarkRoomCallback(BaseCallback):
    """A callback to save learning history during training."""    
    def __init__(
            self, 
            experiment_name: str, 
            save_dir: str = "saved_data/learning_history", 
            verbose: int = 0
        ) -> None:
        super().__init__(verbose)
        self.learning_history_dir = os.path.join(save_dir, experiment_name)
        os.makedirs(self.learning_history_dir, exist_ok=True)
        print(f"Learning history will be saved to: {self.learning_history_dir}")
        self.rollout_episodes = 0
        
    def _on_step(self) -> bool:
        return super()._on_step()
        
    def _on_rollout_end(self) -> None:
        self.rollout_episodes += 1
        savepath = os.path.join(self.learning_history_dir, 
                                f"learning-history_episode-{self.rollout_episodes}.npz")
        observations = self.locals['rollout_buffer'].observations.squeeze()
        actions = self.locals['rollout_buffer'].actions.squeeze()
        rewards = self.locals['rollout_buffer'].rewards
        np.savez(savepath, observations=observations, actions=actions, rewards=rewards)
        return super()._on_rollout_end()

@dataclass
class Args:
    env: SetupDarkRoom
    "environment"
    
    num_envs: int = 5
    "number of envs (number of cpus)"
    
    batch_size: int = 100
    "batch_size = n_steps * num_envs"
    
    lambda_val: float = 0.95
    "gae lambda"
    
    gamma: float = 0.99
    "agent_discount"
    
    entropy_bonus_weight: float = 0.01
    "ent_coef"
    
    optim: Literal["adam", "rms"] = "adam"
    "optimizer"
    
    lr: float = 1e-4
    "learning rate"
    
    mlp_hidden_dim: int = 128
    "mlp hidden dim"
    
    mlp_layers: int = 3
    "mlp"
    
    verbose: int = 1
    "verbosity"
    
    tensorboard_log_dir: str = "saved_data/logs/tb"
    "tensorboard logging directory path"
    
    seed: int = 0
    "seed for training"
    
    seed_eval: int = 100
    "seed for evalutaion"
    
    total_max_timesteps: int = int(1e7)
    "total timesteps"
    
    eval_freq: int = int(5e3)
    "frequency of evaluation in timesteps"
    
    n_eval_episodes: int = 5
    "number of evaluation episodes"
    
    max_no_improvement_evals: int = 100
    "StopTrainingOnNoModelImprovement parameter"
    
    def __post_init__(self):
        self.setup_tensorboard()
        set_random_seed(self.seed)
    
    @property
    def n_steps(self):
        return self.batch_size // self.num_envs
    
    @property
    def use_rms_prop(self):
        if self.optim == 'adam':
            return False
        return True

    @property
    def policy_kwards(self):
        net_arch=dict(
            pi=[self.mlp_hidden_dim for _ in range(self.mlp_layers-2)],  # [128]
            qf=[self.mlp_hidden_dim for _ in range(self.mlp_layers-2)],  # [128]
        )
        return dict(net_arch=net_arch)
    
    @property
    def n_epochs(self):
        return self.total_max_timesteps // self.eval_freq
    
    def setup_tensorboard(self):
        self.tensorboard_log_dir = os.path.join(self.tensorboard_log_dir, self.env.experiment_name)
        os.makedirs(self.tensorboard_log_dir)
        
    def print_args(self):
        print("The following arguments will be used in the experiment:")
        print(yaml.dump(asdict(self)))
    
if __name__ == "__main__":

    # read arguments    
    config = tyro.cli(Args)
    config.print_args()
    
    # setup enviroment
    envs = SubprocVecEnv(config.env.init_n_envs(num_envs=config.num_envs, seed=config.seed))  # training envs
    eval_env = SubprocVecEnv(config.env.init_n_envs(num_envs=config.num_envs, seed=config.seed_eval))  # training envs
    # eval_env = config.env.init_env(seed=config.seed_eval)  # evaluation env

    logger = configure(config.env.monitor_logs_dir, ["stdout", "csv", "tensorboard"])
    model = A2C("MlpPolicy", 
                envs, 
                verbose=config.verbose, 
                learning_rate=config.lr,
                n_steps=config.n_steps,  # batch_size = n_steps * n_envs = 100
                gamma=config.gamma,
                gae_lambda=config.lambda_val,
                ent_coef=config.entropy_bonus_weight,
                use_rms_prop=config.use_rms_prop,  # "adam"
                # tensorboard_log=config.tensorboard_log_dir,
                policy_kwargs=config.policy_kwards,
    )
    model.set_logger(logger)
    learning_hist_callback=LearningHistoryDarkRoomCallback(config.env.experiment_name)
    
    # Setup evaluation
    # Stop training if there is no improvement after more than 100 evaluations
    stop_train_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=config.max_no_improvement_evals, 
        min_evals=config.n_eval_episodes, 
        verbose=config.verbose
    )
    eval_callback = EvalCallback(
        eval_env, 
        eval_freq=config.eval_freq, 
        callback_after_eval=stop_train_callback, 
        verbose=config.verbose
    )
        
    callbacks = CallbackList([learning_hist_callback, eval_callback])
    model.learn(total_timesteps=config.total_max_timesteps, callback=callbacks)
    