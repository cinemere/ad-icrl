# %%
import os
# %%
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
file = files[last_model_id]
file = files[other_model_id]
model_path = os.path.join(experiment_dir, file)
# %%
# model_path = "/home/cinemere/work/repo/ad-icrl/saved_data/saved_models/darkroom_25-Jul-16-45-10/model_7000.pt"
# model_path = "/home/cinemere/work/repo/ad-icrl/saved_data/saved_models/darkroom_25-Jul-23-04-58/model_48000.pt"
# darkroom_27-Jul-11-40-20
model_path = "/home/cinemere/work/repo/ad-icrl/saved_data/saved_models/darkroom_27-Jul-11-40-20/model_16000.pt"
model_path = "/home/cinemere/work/repo/ad-icrl/saved_data/saved_models/darkroom_27-Jul-11-40-20/model_120000.pt"
input_args = "--config.seq-len 100 --config.eval-episodes 100 --config.eval-freq 1000 --config.learning-rate 0.0005 --config.weight-decay 0.001 --config.embedding-dim 64 --config.warmup-steps 1000 --config.debug"

# %%
import tyro
from src.dt.train import TrainConfig

def read_config(config: TrainConfig):
    return config

config = tyro.cli(read_config, args=input_args.split())

# %%
# from src.dt.train import TrainConfig
# from src.collect_data.collect import SetupDarkRoom

# env_config = SetupDarkRoom()
# config = TrainConfig(env_config=env_config) 
# %%
tmp_env = config.env_config.init_env()
# %%
import torch
from src.dt.model import DecisionTransformer
device = torch.device("cuda")
model = DecisionTransformer(
        state_dim=tmp_env.observation_space.n,  # 81
        action_dim=tmp_env.action_space.n,
        seq_len=config.seq_len,  # 100
        embedding_dim=config.embedding_dim,  # 64,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        attention_dropout=config.attention_dropout,  # 0.5,
        embedding_dropout=config.dropout,  # 0.1
        residual_dropout=config.dropout,  # 0.1
).to(device=device)
# %%
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

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
    config.env_config, model, test_goal_idxs, 1000, 
    device=device,
    seed=config.eval_seed + 102)
eval_info_train, debug_info_train = evaluate_in_context(
    config.env_config, model, train_goal_idxs, 1000, 
    device=device,
    seed=config.eval_seed + 102)
# %%
import matplotlib.pyplot as plt
for key, values in eval_info_test.items():
    plt.plot(values, label=key)
    # plt.show()
for key, values in eval_info_train.items():
    plt.plot(values, label=key)
    # plt.show()
plt.legend()    
# %%
# from src.dt.seq_dataset import SequenceDataset
# dataset = SequenceDataset(
#     train_goal_idxs[:3], 
#     seq_len=config.seq_len, 
#     filter_episodes=1,
#     max_episodes=500_000)
# # %%
# dataset.data[0].__dict__
# # %%
# import os
# path_lh1 = "/home/cinemere/work/repo/ad-icrl/saved_data/learning_history/sb-A2C/darkroom_goal=00_11-Jul-23-29-51"
# path_lh2 = "/home/cinemere/work/repo/ad-icrl/saved_data/learning_history/darkroom-goal=00_24-Jul-20-22-22"
# # path_lh2 = "/home/cinemere/work/repo/ad-icrl/saved_data/learning_history/darkroom-goal=00_24-Jul-20-28-35"
# filenames1 = os.listdir(path_lh1)
# filenames2 = os.listdir(path_lh2)
# # %%
# import numpy as np
# path_lh = "/home/cinemere/work/repo/ad-icrl/saved_data/learning_history/ppo/darkroom-goal=00_24-Jul-23-47-57.npz"
# lh1 = np.load(os.path.join(path_lh1, filenames1[0]))
# lh2 = np.load(os.path.join(path_lh2, filenames2[0]))
# lh = np.load(path_lh)
# # %%
# lh1['rewards']
# # %%
# lh2['rewards']
# # %%
# lh['rewards'][2000:2020]
# # %%
# lh['observations']

# # %%
# lh['actions']
# %%
debug_info_test['goal_idxs']
# %%
debug_info_test.keys()
# %%
import numpy as np
def draw_sample_eff_graph(
    eval_res, name, ylim=None, max_return=None, max_return_eps=None
):
    rets = np.vstack([h for h in eval_res.values()])
    means = rets.mean(0)
    stds = rets.std(0)
    x = np.arange(1, rets.shape[1] + 1)

    fig, ax = plt.subplots(dpi=100)
    ax.grid(visible=True)
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    ax.plot(x, means)
    ax.fill_between(x, means - stds, means + stds, alpha=0.2)

    ax.set_ylabel("Return")
    ax.set_xlabel("Episodes In-Context")
    ax.set_title(f"{name}")

    if max_return is not None:
        ax.axhline(
            max_return,
            ls="--",
            color="goldenrod",
            lw=2,
            label=f"optimal_return: {max_return:.2f}",
        )
    if max_return_eps is not None:
        ax.axhline(
            max_return_eps,
            ls="--",
            color="indigo",
            lw=2,
            label=f"max_perf_return: {max_return_eps:.2f}",
        )
    if max_return_eps is not None or max_return is not None:
        plt.legend()
    plt.show()
    # fig.savefig(f"rets_vs_eps_{name}.png")
    # return f"rets_vs_eps_{name}.png"
# %%
from src.collect_data.generate_goals import max_episode_reward

# max_optimal_return = max_episode_reward()
pic_name_train = draw_sample_eff_graph(
    eval_info_train,
    ylim=[-0.3, 20],
    name=f"train; max_perf: 0.5",
    # max_return=max_optimal_return,
    # max_return_eps=max_perf_return,
)
pic_name_train = draw_sample_eff_graph(
    eval_info_test,
    ylim=[-0.3, 20],
    name=f"test; max_perf: 0.5",
    # max_return=max_optimal_return,
    # max_return_eps=max_perf_return,
)

# %%

def draw_grid(grid_size):
    fig, ax = plt.subplots(figsize=(grid_size, grid_size))
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)

    # Major ticks
    ax.set_xticks(np.arange(0, grid_size, 1))
    ax.set_yticks(np.arange(0, grid_size, 1))
    # Labels for major ticks
    ax.set_xticklabels(np.arange(0, grid_size, 1), minor=True)
    ax.set_yticklabels(np.arange(0, grid_size, 1), minor=True)

    # Minor ticks
    ax.set_xticks(np.arange(0.5, grid_size, 1), minor=True)
    ax.set_yticks(np.arange(0.5, grid_size, 1), minor=True)
    ax.tick_params(
        which="major", bottom=False, left=False, labelbottom=False, labelleft=False
    )

    ax.grid(which="major", axis="both", color="k")
    colormesh = ax.pcolormesh(np.zeros((grid_size, grid_size)))

    return fig, ax, colormesh
# %%
size = 9
fig, ax, colormesh = draw_grid(size)
# %%
fig
# %%
ax
# %%
colormesh
# %% PLOTTING OF OUR COOL TABLE WITH % OF BEST RETURNS
import matplotlib.pyplot as plt
from src.collect_data.generate_goals import get_all_goals, max_episode_reward
size = 9
goal_colors = np.zeros((size, size))
for goal_idx, goal in enumerate(get_all_goals(size)):
    if goal_idx in eval_info_train:
        goal_colors[goal[0], goal[1]] = -2
    elif goal_idx in eval_info_test:
        goal_colors[goal[0], goal[1]] = 2

plt.imshow(goal_colors.T, cmap='summer', alpha=0.5)
all_goals = get_all_goals(size)
for goal_idx, rewards in [*eval_info_train.items(), *eval_info_test.items()]:
    x, y = all_goals[goal_idx]
    # reward = rewards[-1] / 20 #/ max_episode_reward(goal_idx)
    reward = np.mean(rewards) / 20 #/ max_episode_reward(goal_idx)
    plt.text(x-0.4, y+0.1, f"{reward:.0%}")
# %%
debug_info_test
# %%
from matplotlib import animation
from functools import partial

def animate_traj(env, states, actions, goal, key_pos=None, name=""):
    size = env.size
    fig, ax, colormesh = draw_grid(size)
    states = env.state_to_pos(states)

    actions_dict = {0: "O", 1: "↓", 2: "→", 3: "↑", 4: "←"}

    def animate(i, goal):
        mesh = np.zeros((size, size))
        mesh[int(states[0][i]), int(states[1][i])] = 1
        # mesh[np.random.randint(0, size), np.random.randint(0, size)] = 1
        mesh[goal[0], goal[1]] = 2

        if key_pos is not None:
            mesh[key_pos[0], key_pos[1]] = 3
        colormesh = ax.pcolormesh(mesh)

        act = actions_dict[actions[i]]
        ax.text(
            states[1][i] + 0.4, states[0][i] + 0.4, act, color="orange", fontsize=20
        )

        return colormesh

    animate = partial(animate, goal=goal)

    anim = animation.FuncAnimation(fig, animate, frames=states.shape[1], blit=False)
    anim.save(f"basic_animation_{name}.gif", fps=2)

# %%
tmp_env.state_to_pos(debug_info_test['states'])
# %%
