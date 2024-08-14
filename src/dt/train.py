# %%
import os
import tyro
from dataclasses import dataclass, asdict, field
import yaml
import itertools
from collections import defaultdict
from typing import Tuple, Optional, List
from tqdm import tqdm
from tqdm.auto import trange

import wandb
from gymnasium.vector import SyncVectorEnv
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F  # noqa
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.collect_data.collect import SetupDarkRoom
from src.dt.seq_dataset2 import SequenceDataset
from src.dt.model import DecisionTransformer
from src.dt.schedule import cosine_annealing_with_warmup
from src.dt.eval import evaluate_in_context

from src.collect_data.generate_goals import max_episode_reward

DEVICE = os.getenv("DEVICE", "cpu")
if "cuda" in DEVICE:
    assert torch.cuda.is_available()

def get_goal_idxs(permutations_file: str = 'saved_data/permutations_9.txt',
                  train_test_split: float = 0.3,
                  debug: bool = False):
    goal_idxs = np.loadtxt(permutations_file, dtype=int).tolist()
    test_size = int(len(goal_idxs) * train_test_split)
    train_idxs, test_idxs = goal_idxs[:-test_size], goal_idxs[-test_size:]
    if debug:
        train_idxs, test_idxs = train_idxs[:10], test_idxs[:10]
    return train_idxs, test_idxs

LEARNING_HISTORY_DIRS = [
    "/home/cinemere/work/repo/ad-icrl/saved_data/learning_history/ppo-01",
    "/home/cinemere/work/repo/ad-icrl/saved_data/learning_history/ppo-02",
    "/home/cinemere/work/repo/ad-icrl/saved_data/learning_history/ppo-03"]

@dataclass
class TrainConfig:
    # ---- logging params ---
    env_config: SetupDarkRoom
    checkpoints_path: Optional[str] = "saved_data/saved_models"
    project: str = 'AD'
    group: str = 'debug'
    entity: str = 'albinakl'
    
    # ---- dataset params ----
    permutations_file: str = 'saved_data/permutations_9.txt'
    "file with permutations of goal idxs"
    train_test_split: float = 0.5
    "percent of test goals"
    filter_episodes: int = 1
    "shrink the dataset by filtering episodes"
    learning_history_dirs: str | List[str] = field(default_factory=LEARNING_HISTORY_DIRS.copy)
    
    # ---- model params ----
    seq_len: int = 60
    "sequence length (steps in env in context)"
    embedding_dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    attention_dropout: float = 0.5
    dropout: float = 0.1
    model_path: str = ""
    
    # ---- optimizer params ----
    learning_rate: float = 3e-4  # 1e-4
    weight_decay: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    warmup_steps: int = 5_000 # 10_000
    # warmup_ratio: float = 0.1
    clip_grad: Optional[float] = 1.0

    batch_size: int = 128  # 512
    num_updates: int = 300_000
    "batch size (dataloader param)"    
    num_workers: int = 1
    "num_workers (dataloader param)"

    # ---- eval ----
    eval_freq: int = 1000
    eval_episodes: int = 10
    eval_seed: int = 0

    # ---- debug ----
    debug: bool = False
    
    def __post_init__(self):
        if self.debug:
            # self.filter_episodes = 10
            ...
            
        if self.checkpoints_path is not None:
            path = os.path.join(self.checkpoints_path, self.env_config.experiment_name)
            os.makedirs(path, exist_ok=True)
            self.checkpoints_path = path
            
    def save_args(self):
        config_file_path = os.path.join(self.checkpoints_path, "config.yaml")
        with open(config_file_path, "w") as config_file:
            config_file.write(yaml.safe_dump(asdict(self)))


def train(config: TrainConfig):
    config.save_args()
    
    wandb.init(entity=config.entity,
               project=config.project, 
               group=config.group, 
               name=config.env_config.experiment_name, 
               config=asdict(config))
    
    train_goal_idxs, test_goal_idxs = get_goal_idxs(
        permutations_file=config.permutations_file, 
        train_test_split=config.train_test_split,
        debug=config.debug)
    
    dataset = SequenceDataset(goal_idxs=train_goal_idxs, 
                              seq_len=config.seq_len, 
                              filter_episodes=config.filter_episodes,
                              learning_history_dirs=config.learning_history_dirs)

    dataloader = DataLoader(dataset,
                            batch_size=config.batch_size,
                            pin_memory=True,
                            num_workers=config.num_workers)

    device = torch.device(DEVICE)

    tmp_env = config.env_config.init_env()
    model = DecisionTransformer(
        state_dim=tmp_env.observation_space.n, # 81
        action_dim=tmp_env.action_space.n,
        seq_len=config.seq_len,
        embedding_dim=config.embedding_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        attention_dropout=config.attention_dropout,
        residual_dropout=config.dropout,
        embedding_dropout=config.dropout,
    ).to(device)
    if config.model_path:
        model.load_state_dict(torch.load(config.model_path, map_location=device))


    optim = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=config.betas,
    )
    
    # scheduler = CosineAnnealingLR(
    #     optimizer=optim,
    #     eta_min=1e-6,
    #     T_max=config.num_updates,   
    # )
    scheduler = cosine_annealing_with_warmup(
        optimizer=optim,
        warmup_steps=config.warmup_steps,
        total_steps=config.num_updates,
    )
    # scaler = torch.cuda.amp.GradScaler()
    
    dataloader_iter = iter(dataloader)
    for step in trange(config.num_updates, desc="Training"):
        batch = next(dataloader_iter)
        # print(batch)
        states, actions, rewards = [b.to(device) for b in batch]

        # with torch.cuda.amp.autocast():
        predicted_actions = model(
            states=states,
            actions=actions,
            rewards=rewards)
        loss  = F.cross_entropy(
            predicted_actions.flatten(0, 1),
            actions.detach().flatten(0, 1))

        # scaler.scale(loss).backward()
        # if config.clip_grad is not None:
        #     scaler.unscale_(optim)
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
        # scaler.step(optim)
        # scaler.update()
        # optim.zero_grad(set_to_none=True)
        # scheduler.step()
         
        optim.zero_grad()
        loss.backward()
        if config.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
        optim.step()
        scheduler.step()
        
        with torch.no_grad():
            a = torch.argmax(predicted_actions.flatten(0, 1), dim=-1)
            t = actions.flatten()
            accuracy = torch.sum(a == t) / (config.batch_size * config.seq_len)

        wandb.log(
            {
                "loss": loss.item(),
                "accuracy": accuracy,
                "step": step,
                "lr": scheduler.get_last_lr()[0],
            },
            step=step,
        )
        
        if step % config.eval_freq == 0 or step == config.num_updates - 1:
            model.eval()
            eval_info_train, debug_info_train = evaluate_in_context(config.env_config, 
                                                        model, train_goal_idxs, 
                                                        config.eval_episodes, 
                                                        device, config.eval_seed)
            eval_info_test, debug_info_test = evaluate_in_context(config.env_config, 
                                                        model, test_goal_idxs, 
                                                        config.eval_episodes, 
                                                        device, config.eval_seed)            
            print("eval train:\n")
            for goal_idx, logged_returns in eval_info_train.items():
                print("goal:", goal_idx, 
                      "max reward:", max_episode_reward(goal_idx),
                      logged_returns)
            # print(*eval_info_train.items(), sep="\n")
            print("eval test:\n")
            for goal_idx, logged_returns in eval_info_test.items():
                print("goal:", goal_idx, 
                      "max reward:", max_episode_reward(goal_idx),
                      logged_returns)
            # print(*eval_info_test.items(), sep="\n")
        
            model.train()
            wandb.log(
                {
                    "eval/train_goals/mean_return": np.mean(
                        [h[-1] for h in eval_info_train.values()]
                    ),
                    "eval/train_goals/median_return": np.median(
                        [h[-1] for h in eval_info_train.values()]
                    ),
                    "eval/test_goals/mean_return": np.mean(
                        [h[-1] for h in eval_info_test.values()]
                    ),
                    "eval/test_goals/median_return": np.median(
                        [h[-1] for h in eval_info_test.values()]
                    ),
                    # "eval/train_goals/graph": wandb.Image(pic_name_train),
                    # "eval/test_goals/graph": wandb.Image(pic_name_test),
                    # "eval/train_goals/video": wandb.Video(
                    #     "basic_animation_train.gif"
                    # ),
                    # "eval/test_goals/video": wandb.Video(
                    #     "basic_animation_test.gif"
                    # ),
                    "epoch": step,
                },
                step=step,
            )
            if config.checkpoints_path is not None:
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        config.checkpoints_path, f"model_{step}.pt"
                    ),
                )
                torch.save(
                    optim.state_dict(),
                    os.path.join(
                        config.checkpoints_path, f"optim_{step}.pt"
                    ),
                )
                torch.save(
                    scheduler.state_dict(),
                    os.path.join(
                        config.checkpoints_path, f"scheduler_{step}.pt"
                    ),
                )
                torch.save(
                    debug_info_train, 
                    os.path.join(
                        config.checkpoints_path, f"debug_info_train_{step}.pt"
                    ),
                )
                torch.save(
                    debug_info_test, 
                    os.path.join(
                        config.checkpoints_path, f"debug_info_test_{step}.pt"
                    ),
                )
                
    if config.checkpoints_path is not None:
        torch.save(
            model.state_dict(), os.path.join(config.checkpoints_path, f"model_last.pt")
        )
        
if __name__ == "__main__":
    tyro.cli(train)