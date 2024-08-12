import os
import tyro
import numpy as np

def get_all_goals(size: int = 9):
    return np.mgrid[0:size, 0:size].reshape(2, -1).T 

def max_episode_reward(goal_idx: int, episode_lenght: int = 20, size: int = 9):
    center_pos = (size // 2, size // 2)
    goal_pos = get_all_goals(size)[goal_idx]
    x_steps = abs(center_pos[0] - goal_pos[0])
    y_steps = abs(center_pos[1] - goal_pos[1])
    return min(episode_lenght - x_steps - y_steps + 1, episode_lenght)

def save_goals(
        size: int = 9,
        save_dir: str = "saved_data/",
    ):
    goals = get_all_goals(size)
    permutation = np.random.permutation(len(goals))
    
    # save outputs
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    np.savetxt(os.path.join(save_dir, f"goals_{size}.txt"), goals, fmt="%d")
    np.savetxt(os.path.join(save_dir, f"permutations_{size}.txt"), permutation, fmt="%d")
    
if __name__ == "__main__":
    tyro.cli(save_goals)