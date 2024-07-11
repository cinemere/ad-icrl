import os
import tyro
import numpy as np

def get_all_goals(size: int = 9):
    return np.mgrid[0:size, 0:size].reshape(2, -1).T 

def save_goals(
        size: int = 9,
        goals_file: str = "saved_data/goals.txt",
        permutations_file: str = "saved_data/permutations.txt",
    ):
    goals = get_all_goals(size)
    permutation = np.random.permutation(len(goals))
    
    # save goals
    os.makedirs(os.path.dirname(goals_file), exist_ok=True)
    np.savetxt(goals_file, goals, fmt="%d")
    
    # save permutations
    os.makedirs(os.path.dirname(permutations_file), exist_ok=True)
    np.savetxt(permutations_file, permutation, fmt="%d")

if __name__ == "__main__":
    tyro.cli(save_goals)