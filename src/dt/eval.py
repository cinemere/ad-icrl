import itertools
from collections import defaultdict
from typing import List, Literal
from tqdm import tqdm

import numpy as np
import torch
from gymnasium.vector import SyncVectorEnv

from src.data.env import SetupDarkRoom
from src.dt.model import DecisionTransformer

@torch.no_grad()
def evaluate_in_context(env_config: SetupDarkRoom, 
                        model: DecisionTransformer, 
                        goal_idxs: List[int], 
                        eval_episodes: int, 
                        device: torch.DeviceObjType, 
                        seed: int | None = None,
                        mode: Literal["mode", "sample"] = "mode"):
    vec_env = SyncVectorEnv(
        [lambda goal_idx=goal_idx: env_config.get_cls()(goal_index=goal_idx,
                                              enable_monitor_logs=False).init_env()
        for goal_idx in goal_idxs])

    states = torch.zeros(
        (model.seq_len, vec_env.num_envs), dtype=torch.long, device=device
    )
    actions = torch.zeros(
        (model.seq_len, vec_env.num_envs), dtype=torch.long, device=device
    )
    rewards = torch.zeros(
        (model.seq_len, vec_env.num_envs), dtype=torch.long, device=device
    )

    # to track number of episodes for each goal and returns
    num_episodes = np.zeros(vec_env.num_envs)
    returns = np.zeros(vec_env.num_envs)
    # for logging
    eval_info = defaultdict(list)
    pbar = tqdm(total=vec_env.num_envs * eval_episodes, position=1)

    state, _ = vec_env.reset(seed=seed)
    for step in itertools.count(start=1):
        # roll context back for new step
        states = states.roll(-1, dims=0)
        actions = actions.roll(-1, dims=0)
        rewards = rewards.roll(-1, dims=0)
        # set current state
        states[-1] = torch.tensor(state, device=device)

        # predict next action,
        logits = model(
            states=states[-step:].permute(1, 0),
            actions=actions[-step:].permute(1, 0),
            rewards=rewards[-step:].permute(1, 0),
        )[0][:, -1]  # )[:, -1]
        dist = torch.distributions.Categorical(logits=logits)
        
        if mode == 'mode':  
            action = dist.mode
        else:
            action = dist.sample()

        # query the world
        state, reward, terminated, truncated, _ = vec_env.step(action.cpu().numpy())
        done = terminated | truncated

        actions[-1] = action
        rewards[-1] = torch.tensor(reward, device=device)

        num_episodes += done.astype(int)
        returns += reward

        # log returns if done
        for i, d in enumerate(done):
            if d and num_episodes[i] <= eval_episodes:
                eval_info[goal_idxs[i]].append(returns[i])
                # reset return for this goal
                returns[i] = 0.0
                # update tqdm
                pbar.update(1)

        # check that all goals are done
        if np.all(num_episodes > eval_episodes):
            break

    debug_info = {"states": states, "actions": actions, "goal_idxs": goal_idxs}

    return eval_info, debug_info