# %%
import os
experiment_dir="/home/cinemere/work/repo/ad-icrl/saved_data/saved_models/darkroom_18-Jul-11-13-38"
files = {int(x.split("_")[1].split(".")[0]): x for x in os.listdir(experiment_dir)}
# %%
files
# %%
files = dict(sorted(files.items()))
# %%
last_model_id = list(files.keys())[-1]
other_model_id = 15000
# %%
# file = files[last_model_id]
file = files[other_model_id]
# %%
model_path = os.path.join(experiment_dir, file)
# %%
from src.dt.train import TrainConfig
from src.collect_data.collect import SetupDarkRoom

env_config = SetupDarkRoom()
config = TrainConfig(env_config=env_config) 
# %%
tmp_env = config.env_config.init_env()
# %%
import torch
from src.dt.model import DecisionTransformer
device = torch.device("cuda")
model = DecisionTransformer(
        state_dim=tmp_env.observation_space.n, # 81
        action_dim=tmp_env.action_space.n,
        seq_len=config.seq_len,
        embedding_dim=config.embedding_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
).to(device=device)
# %%
model.load_state_dict(torch.load(model_path, map_location=device))
# %%
model
# %%
from src.dt.train import get_goal_idxs

train_goal_idxs, test_goal_idxs = get_goal_idxs(
    permutations_file=config.permutations_file, 
    train_test_split=config.train_test_split,
    debug=config.debug)
# %%
from src.dt.eval import evaluate_in_context

eval_info_test, debug_info_test = evaluate_in_context(
    config.env_config, model, test_goal_idxs, 1000, config.eval_seed)
# %%
eval_info_test, debug_info_test = evaluate_in_context(
    config.env_config, model, train_goal_idxs, 100, config.eval_seed)
# %%
eval_info_test
# %%
import matplotlib.pyplot as plt
for key, values in eval_info_test.items():
    plt.plot(values, label=key)
plt.legend()    
# %%
from src.dt.seq_dataset import SequenceDataset
dataset = SequenceDataset(
    train_goal_idxs[:3], 
    seq_len=config.seq_len, 
    filter_episodes=1,
    max_episodes=500_000)
# %%
dataset.data[0].__dict__
# %%
import os
path_lh1 = "/home/cinemere/work/repo/ad-icrl/saved_data/learning_history/sb-A2C/darkroom_goal=00_11-Jul-23-29-51"
path_lh2 = "/home/cinemere/work/repo/ad-icrl/saved_data/learning_history/darkroom-goal=00_24-Jul-20-22-22"
# path_lh2 = "/home/cinemere/work/repo/ad-icrl/saved_data/learning_history/darkroom-goal=00_24-Jul-20-28-35"
filenames1 = os.listdir(path_lh1)
filenames2 = os.listdir(path_lh2)
# %%
import numpy as np
path_lh = "/home/cinemere/work/repo/ad-icrl/saved_data/learning_history/ppo/darkroom-goal=00_24-Jul-23-47-57.npz"
lh1 = np.load(os.path.join(path_lh1, filenames1[0]))
lh2 = np.load(os.path.join(path_lh2, filenames2[0]))
lh = np.load(path_lh)
# %%
lh1['rewards']
# %%
lh2['rewards']
# %%
lh['rewards'][2000:2020]
# %%
lh['observations']

# %%
lh['actions']
# %%
