# %% ------------------------------------------

import os
from typing import List, Dict, Any
import yaml
import tyro
from dataclasses import asdict, dataclass
from functools import cached_property

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.dt.train import TrainConfig, get_goal_idxs
from src.collect_data.collect import SetupDarkRoom
from src.dt.utils import set_seed
from src.dt.eval import evaluate_in_context
from src.dt.model import DecisionTransformer

@dataclass
class EvalConfig:
    # --- init ---
    model_dir: str  # directory with models (e.g. 'model_{n_steps}.pt')
    out_dir: str | None = None  # directory where to save outputs (othervise `model_dir` will be used)
    
    # --- call ---
    seed: int = 0 # seed the evaluation
    n_repeats: int = 5 # number of evaluations
    eval_episodes: int = 100 # number of episodes to evaluate
    only_last: bool = True # eval only the latest model in the dir

class Evaluator:
    def __init__(self, model_dir: str, out_dir: str | None = None):
        self.model_dir = model_dir
        self.out_dir = os.path.join(self.model_dir if out_dir is None else out_dir, "eval_plots")
        os.makedirs(self.out_dir, exist_ok=True)
        print(f"Output plots will be saved to {self.out_dir}")

    @cached_property
    def experiment_name(self) -> str:
        return self.model_dir.rstrip(os.sep).split(os.sep)[-1]

    @cached_property
    def model_paths(self) -> Dict[int, str]:
        """get model paths

        Returns:
            Dict[int, str]: number of steps, path to the model
        """
        models = [name for name in os.listdir(self.model_dir) if "model" in name]
        models = {int(name.rstrip(".pt").split("_")[-1]) : os.path.join(self.model_dir, name) for name in models}
        models = dict(sorted(models.items()))
        return models
    
    @cached_property
    def config(self) -> Dict[str, Any]:
        cfg_path = [name for name in os.listdir(self.model_dir) if "config" in name]
        print(cfg_path)
        assert len(cfg_path) == 1
        cfg_path = os.path.join(self.model_dir, cfg_path[0])
        with open(cfg_path, "r") as cfg_file:
            config = yaml.safe_load(cfg_file.read().strip())
        config['env_config'] = SetupDarkRoom(**config['env_config'])
        config = TrainConfig(**config)
        return config

    def __call__(self, 
                 seed: int = 0, 
                 n_repeats: int = 5, # number of evaluations
                 eval_episodes: int = 100, # number of episodes to evaluate
                 only_last: bool = True, # eval only the latest model in the dir
        ) -> None:
        
        # initialize_model
        train_goal_idxs, test_goal_idxs = get_goal_idxs(
            permutations_file=self.config.permutations_file, 
            train_test_split=self.config.train_test_split,
            debug=self.config.debug)

        device = torch.device('cpu')

        tmp_env = self.config.env_config.init_env()
        model = DecisionTransformer(
            state_dim=tmp_env.observation_space.n, # 81
            action_dim=tmp_env.action_space.n,
            seq_len=self.config.seq_len,
            embedding_dim=self.config.embedding_dim,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            attention_dropout=self.config.attention_dropout,
            residual_dropout=self.config.residual_dropout,
            embedding_dropout=self.config.embedding_dropout,
        ).to(device)
        
        for n_steps, model_path in reversed(self.model_paths.items()):
    
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            eval_info_train, eval_info_test = [], []

            for n_repeat in range(n_repeats):
                set_seed(seed + n_repeat)            

                _eval_info_train, _ = evaluate_in_context(self.config.env_config, model, train_goal_idxs, eval_episodes, device)
                _eval_info_test, _ = evaluate_in_context(self.config.env_config, model, test_goal_idxs, eval_episodes, device)

                # # for debug:
                # _eval_info_train = defaultdict(list)
                # _eval_info_test = defaultdict(list)
                # for key in train_goal_idxs:
                #     _eval_info_train[key] = ((np.random.rand(eval_episodes) * 10).astype(int) + np.arange(eval_episodes)).tolist()
                # for key in test_goal_idxs:
                #     _eval_info_test[key] = ((np.random.rand(eval_episodes) * 10).astype(int) + np.arange(eval_episodes)).tolist()
            
                eval_info_train.append(_eval_info_train)
                eval_info_test.append(_eval_info_test)
            
            eval_info_train = np.vstack([np.mean( [x[key] for x in eval_info_train], axis=0) \
                for key in eval_info_train[0].keys()])
            
            eval_info_test = np.vstack([np.mean( [x[key] for x in eval_info_test], axis=0) \
                for key in eval_info_test[0].keys()])
            
            plt.errorbar(np.arange(eval_info_train.shape[1]), 
                        np.mean(eval_info_train, axis=0),
                        np.std(eval_info_train, axis=0), label=f'train ({eval_info_train.shape[0]} tasks)')
            plt.errorbar(np.arange(eval_info_test.shape[1]), 
                        np.mean(eval_info_test, axis=0),
                        np.std(eval_info_test, axis=0), label=f'test ({eval_info_test.shape[0]} tasks)')
            plt.xlabel("number of episode")
            plt.ylabel("mean episode return")
            plt.legend()
            plt.title(f"In-context learning on {self.experiment_name} model \n({n_steps} learning steps, {n_repeats} seeds)")
            plt.savefig(os.path.join(self.out_dir, f"mean_eval_{self.experiment_name}_{n_steps}_{seed=}.png"))
            plt.show()

            if only_last:
                break

# %%
# tmp_path = "/home/cinemere/work/repo/ad-icrl/saved_data/saved_models/darkroom_14-Aug-01-33-50"
# ev = Evaluator(tmp_path, "tmp_dir")
# ev.experiment_name
# ev.model_paths
# config = ev.config
# ev(seed=0, n_repeats=2, eval_episodes=5, only_last=True)

# %%


if __name__ == "__main__":
    config = tyro.cli(EvalConfig)
    
    evaluator = Evaluator(
        model_dir=config.model_dir,
        out_dir=config.out_dir, 
    )
    
    evaluator(
        seed=config.seed,
        n_repeats=config.n_repeats,
        eval_episodes=config.eval_episodes,
        only_last=config.only_last,
    )