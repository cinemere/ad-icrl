# %%
import os
import tyro
from dataclasses import dataclass, asdict
import itertools
from collections import defaultdict
from typing import Tuple, Optional
from tqdm import tqdm
from tqdm.auto import trange

import wandb
from gymnasium.vector import SyncVectorEnv
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F  # noqa

from src.collect_data.collect import SetupDarkRoom
from src.dt.seq_dataset import SequenceDataset
from src.dt.model import DecisionTransformer
from src.dt.schedule import cosine_annealing_with_warmup
from src.dt.eval import evaluate_in_context

DEVICE = os.getenv("DEVICE", "cpu")
if "cuda" in DEVICE:
    assert torch.cuda.is_available()

def get_goal_idxs(permutations_file: str = 'saved_data/permutations.txt',
                  train_test_split: float = 0.3,
                  debug: bool = False):
    goal_idxs = np.loadtxt(permutations_file, dtype=int)
    test_size = int(len(goal_idxs) * train_test_split)
    train_idxs, test_idxs = goal_idxs[:-test_size], goal_idxs[-test_size:]
    if debug:
        train_idxs, test_idxs = train_idxs[:3], test_idxs[:3]
    return train_idxs, test_idxs

@dataclass
class TrainConfig:
    # ---- logging params ---
    env_config: SetupDarkRoom
    checkpoints_path: Optional[str] = "saved_data/saved_models"
    project: str = 'AD'
    group: str = 'debug'
    entity: str = 'albinakl'
    
    # ---- dataset params ----
    permutations_file: str = 'saved_data/permutations.txt'
    "file with permutations of goal idxs"
    train_test_split: float = 0.3
    "percent of test goals"
    filter_episodes: int = 1
    "shrink the dataset by filtering episodes"
    
    # ---- model params ----
    seq_len: int = 60
    "sequence length (steps in env in context)"
    embedding_dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    # ---- optimizer params ----
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    warmup_steps: int = 1000 # 10_000
    warmup_ratio: float = 0.1
    clip_grad: Optional[float] = 1.0

    batch_size: int = 8  # 512
    num_updates: int = 300_000
    "batch size (dataloader param)"    
    num_workers: int = 1
    "num_workers (dataloader param)"

    # ---- eval ----
    eval_freq: int = 1000
    eval_episodes: int = 2
    eval_seed: int = 0

    # ---- debug ----
    debug: bool = True
    
    def __post_init__(self):
        if self.debug:
            self.filter_episodes = 10
            
        if self.checkpoints_path:
            path = os.path.join(self.checkpoints_path, self.env_config.experiment_name)
            os.makedirs(path, exist_ok=True)
            self.checkpoints_path = path

def train(config: TrainConfig):
    
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
                              filter_episodes=config.filter_episodes)

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
    ).to(device)

    optim = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=config.betas,
    )
 
    scheduler = cosine_annealing_with_warmup(
        optimizer=optim,
        warmup_steps=config.warmup_steps,
        total_steps=config.num_updates,
    )
    scaler = torch.cuda.amp.GradScaler()
    
    dataloader_iter = iter(dataloader)
    for step in trange(config.num_updates, desc="Training"):
        batch = next(dataloader_iter)
        # print(batch)
        states, actions, rewards = [b.to(device) for b in batch]

        with torch.cuda.amp.autocast():
            predicted_actions = model(
                states=states,
                actions=actions,
                rewards=rewards)
            loss  = F.cross_entropy(
                predicted_actions.flatten(0, 1),
                actions.detach().flatten(0, 1))

        scaler.scale(loss).backward()
        if config.clip_grad is not None:
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
        scaler.step(optim)
        scaler.update()
        optim.zero_grad(set_to_none=True)
        scheduler.step()
         
        # optim.zero_grad()
        # loss.backward()
        # # if config.clip_grad is not None:
        #     # torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
        # optim.step()
        # scheduler.step()
        
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
            # print(f"{eval_info_train=}\n"
            #       f"{debug_info_train=}\n"
            #       f"{eval_info_test=}\n"
            #       f"{debug_info_test=}\n")
        
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
                
    if config.checkpoints_path is not None:
        torch.save(
            model.state_dict(), os.path.join(config.checkpoints_path, f"model_last.pt")
        )
        
if __name__ == "__main__":
    tyro.cli(train)