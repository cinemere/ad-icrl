# ad-icrl
Implementation of paper "In-context Reinforcement Learning with Algorithm Distillation"

## Installation 🧑‍🔧

```bash
git clone ...
cd ad-icrl
python -m venv venv
pip install -r requirements/requirements.txt
```
Set up environmental variable `PYTHONPATH`:

```bash
export PYTHONPATH=.
```

## Repository structure 

```text
.
├── notebooks
│   ├── colab.ipynb
│   ├── config_paper.py  # hyperparams from paper
│   ├── explore_dataset.ipynb  # overview the learning histories
│   └── test_dark_room.py
├── README.md
├── requirements
│   ├── requirements_colab.txt  # requirements for colab
│   └── requirements.txt
├── saved_data
│   ├── learning_history/
│   ├── logs/
│   ├── goals_9.txt
│   └── permutations_9.txt
├── scripts
│   ├── collect_data.sh
│   ├── eval.sh
│   └── train_ad.sh
├── src
│   ├── check_in_context
│   ├── data
│   ├── dt
│   ├── __init__.py
│   └── __pycache__
├── static
│   ├── learning_histories_ppo-01.png
│   ├── learning_histories_ppo-02.png
│   ├── learning_histories_ppo-03.png
│   ├── mean_eval_darkroom_13-Aug-15-09-45_299999_seed=0_mode.png
│   ├── mean_eval_darkroom_13-Aug-15-09-45_299999_seed=0_sampling.png
│   └── mean_eval_darkroom_14-Aug-19-32-05_40000_seed=0.png
└── wandb
    ├── debug-internal.log -> run-20240815_132305-cl5uhidt/logs/debug-internal.log
    ├── debug.log -> run-20240815_132305-cl5uhidt/logs/debug.log
    ├── latest-run -> run-20240815_132305-cl5uhidt
    └── run-20240815_132305-cl5uhidt
```


## Quick start 🏃

### 0. init on wandb

...

### 1. Generate goals and permuations 

**or** use the provided file [saved_data/permutations_9.txt]().

```python
python src/data/generate_goals.py
```

### 2. Learn PPO agents to collect the dataset

**or** load the trajectories from gdrive via [link](https://drive.google.com/drive/folders/1_pExW9O4SoaraeDZCu05xageE2HBFj_d?usp=sharing).

Run the script to create the trajectories:

```python
chmod +x ./scripts/collect_data.sh
./scripts/collect_data.sh
```

The script will use `src/data/ppo.py` file. This script uses `wandb` logging, to disable it provide `--no-track` flag.

Observe the learning process on tensorboard:
```bash
tensorboard --logdir saved_data/logs/
```

### 3. Train AD


```bash
python src/dt/train.py
```