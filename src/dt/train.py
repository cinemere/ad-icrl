# %%
import numpy as np
permutations_file = 'saved_data/permutations.txt'
goal_idxs = np.loadtxt(permutations_file, dtype=int)
train_test_split = 0.3
test_size = int(len(goal_idxs) * train_test_split)

train_goals, test_goals = goal_idxs[:-test_size], goal_idxs[-test_size:]

# %%
from src.dt.seq_dataset import SequenceDataset
train_dataset = SequenceDataset(train_goals[:3], seq_len=100, filter_episodes=10)
test_dataset = SequenceDataset(test_goals[:3], seq_len=100, filter_episodes=10)

# %%
from torch.utils.data import DataLoader

batch_size = 8
num_workers = 1

trainloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
    )
testloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
    )
# %%
from src.collect_data.collect import SetupDarkRoom
setup_env = SetupDarkRoom(goal_index=test_goals[0])
# %%
from src.dt.utils import wrap_env

eval_env = wrap_env(
    setup_env.init_env(), 
    # actually it is not right to take these scaling from whole dataset
    # we can use only context for it ...
    state_mean=test_dataset.data[test_goals[0]].state_mean,
    state_std=test_dataset.data[test_goals[0]].state_std,
    reward_scale=test_dataset.data[test_goals[0]].reward_scale,
)

# %%
import torch
from src.dt.model import DecisionTransformer

device = torch.device('cpu')

dt = DecisionTransformer(
    state_dim=eval_env.observation_space.n, # 81
    action_dim=eval_env.action_space.n,
    seq_len=100,
    episode_len=20,
    embedding_dim=128,
    num_layers=4,
    num_heads=8,
).to(device)
# %%
optim = torch.optim.AdamW(
        dt.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
        betas=(0.9, 0.999),
)
warmup_steps = 1000 # 10_000 
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optim,
    lambda steps: min((steps + 1) / warmup_steps, 1),
)
# %%
trainloader_iter = iter(trainloader)
# %%
batch = next(trainloader_iter)
# %%
states, actions, returns, time_steps = [b.to(device) for b in batch]
# %%

# # True value indicates that the corresponding key value will be ignored
# padding_mask = ~mask.to(torch.bool)

predicted_actions = dt(
    states=states,
    actions=actions,
    returns_to_go=returns,
    time_steps=time_steps,
)
# %%
loss = F.mse_loss(predicted_actions, actions.detach(), reduction="none")
# [batch_size, seq_len, action_dim] * [batch_size, seq_len, 1]
loss = (loss * mask.unsqueeze(-1)).mean()

optim.zero_grad()
loss.backward()
if config.clip_grad is not None:
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
optim.step()
scheduler.step()
