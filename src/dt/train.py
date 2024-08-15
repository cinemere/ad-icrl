import os
import tyro
from dataclasses import dataclass, asdict, field
import yaml
from typing import Tuple, Optional, List, Literal
from tqdm.auto import trange

import wandb
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F  # noqa

from src.data.env import SetupDarkRoom
from src.dt.seq_dataset import SequenceDataset
from src.dt.model import DecisionTransformer
from src.dt.schedule import cosine_annealing_with_warmup
from src.dt.eval import evaluate_in_context

from src.data.generate_goals import max_episode_reward

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
    "saved_data/learning_history/ppo-01",
    "saved_data/learning_history/ppo-02",
    "saved_data/learning_history/ppo-03"]

@dataclass
class TrainConfig:
    # ---- logging params ---
    env_config: SetupDarkRoom  # Configuration for the environment setup
    checkpoints_path: Optional[str] = "saved_data/saved_models"  # Path to save model checkpoints
    project: str = 'AD'  # Name of the project
    group: str = 'debug'  # Group name for organizing runs
    entity: str = 'albinakl'  # Entity name for tracking experiments

    # ---- dataset params ----
    permutations_file: str = 'saved_data/permutations_9.txt'  # File with permutations of goal indices
    train_test_split: float = 0.5  # Percentage of data to be used for testing
    filter_episodes: int = 1  # Number of episodes to filter from the dataset
    learning_history_dirs: str | List[str] = field(default_factory=LEARNING_HISTORY_DIRS.copy)  # Directories to load learning histories from

    # ---- model params ----
    seq_len: int = 60  # Length of the input sequence (number of steps in the environment)
    embedding_dim: int = 64  # Dimensionality of the embedding layer
    hidden_dim: int = 256  # Dimensionality of the hidden layers
    num_layers: int = 4  # Number of layers in the model
    num_heads: int = 4  # Number of attention heads
    attention_dropout: float = 0.5  # Dropout rate for the attention layer
    residual_dropout: float = 0.1  # Dropout rate for the residual connections
    embedding_dropout: float = 0.1  # Dropout rate for the embedding layer
    ln_placem: Literal["postnorm", "prenorm"] = "postnorm"  # Placement of Layer Normalization
    add_reward_head: bool = False  # Flag to add a reward prediction head to the model
    load_from_checkpoint: Optional[str] = ""  # Path to load a model from a checkpoint
    
    # ---- optimizer params ----
    learning_rate: float = 3e-4  # Learning rate for the optimizer
    weight_decay: float = 1e-4  # Weight decay for regularization
    betas: Tuple[float, float] = (0.9, 0.999)  # Coefficients used for computing running averages in Adam optimizer
    warmup_steps: int = 5_000  # Number of warmup steps for learning rate scheduling  # 1000
    clip_grad: Optional[float] = 1.0  # Gradient clipping value to prevent exploding gradients

    # ---- dataloader params ---- 
    batch_size: int = 128  # Number of samples per batch for the dataloader # 512
    num_updates: int = 300_000  # Total number of updates to perform during training
    num_workers: int = 1  # Number of worker threads for data loading

    # ---- eval ----
    eval_freq: int = 1000  # Frequency of evaluation during training (in updates)
    eval_episodes: int = 10  # Number of episodes to evaluate during each evaluation step
    eval_seed: int = 0  # Seed for random number generation during evaluation
    mode: Literal["mode", "sample"] = "mode"  # Use argmax over model outputs or sample

    # ---- debug ----
    debug: bool = False  # Flag to enable or disable debug mode
   
    def __post_init__(self):
        if self.debug:
            # debug will use only 10 train and 10 test goals
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
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        attention_dropout=config.attention_dropout,
        residual_dropout=config.residual_dropout,
        embedding_dropout=config.embedding_dropout,
        ln_placem=config.ln_placem,
        add_reward_head=config.add_reward_head,
    ).to(device)
    if config.load_from_checkpoint:
        model.load_state_dict(torch.load(config.load_from_checkpoint, map_location=device))

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
    
    dataloader_iter = iter(dataloader)
    for step in trange(config.num_updates, desc="Training"):
        batch = next(dataloader_iter)
        states, actions, rewards = [b.to(device) for b in batch]

        predicted_actions, predicted_rewards = model(
            states=states,
            actions=actions,
            rewards=rewards)
        loss = F.cross_entropy(
            predicted_actions.flatten(0, 1),
            actions.detach().flatten(0, 1))
        loss_rewards = 0 if predicted_rewards is None else \
        F.binary_cross_entropy_with_logits(
            predicted_rewards.flatten(0, 1),
            rewards.detach().flatten(0, 1).float())
         
        optim.zero_grad()
        (loss + loss_rewards).backward()
        if config.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
        optim.step()
        scheduler.step()
        
        wandb_log = {
            "step": step,
            "lr": scheduler.get_last_lr()[0],
        }
        
        with torch.no_grad():
            a = torch.argmax(predicted_actions.flatten(0, 1), dim=-1)
            t = actions.flatten()
            accuracy = torch.sum(a == t) / (config.batch_size * config.seq_len)
            wandb_log['accuracy'] = accuracy
            wandb_log['loss'] = loss.item()
            
            if predicted_rewards is not None:
                r = (predicted_rewards.flatten() > 0.5).long()
                t = rewards.flatten()
                accuracy_reward = torch.sum(r == t) / (config.batch_size * config.seq_len)
                wandb_log['accuracy_reward'] = accuracy_reward
                wandb_log['loss_reward'] = loss_rewards.item()

        wandb.log(wandb_log, step=step)
        
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
            print("eval test:\n")
            for goal_idx, logged_returns in eval_info_test.items():
                print("goal:", goal_idx, 
                      "max reward:", max_episode_reward(goal_idx),
                      logged_returns)
        
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
                
    if config.checkpoints_path is not None:
        torch.save(
            model.state_dict(), os.path.join(config.checkpoints_path, f"MODEL_last.pt")
        )
        
if __name__ == "__main__":
    tyro.cli(train)