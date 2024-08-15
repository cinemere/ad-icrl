# ad-icrl
Implementation of [paper](https://arxiv.org/abs/2210.14215) "In-context Reinforcement Learning with Algorithm Distillation"

## Installation 🧑‍🔧

```bash
git clone ...
cd ad-icrl
python -m venv venv
pip install -r requirements/requirements.txt
```

## Environment Variables

Set up environmental variable `PYTHONPATH` **is required** to run the project:

```bash
export PYTHONPATH=.
```

Also you can set up:

```DEVICE```

## Repository structure 

```text
.
├── notebooks/
│   ├── colab.ipynb                        # run training in colab
│   ├── explore_dataset.ipynb              # overview the dataset for training
│   ├── config_paper.py                    # hyperparams from paper
│   └── test_dark_room.py                  # playground for dark room env
├── README.md
├── requirements/
│   ├── requirements_colab.txt
│   └── requirements.txt
├── saved_data                             # dir where to save data during execution
│   ├── goals_9.txt
│   ├── learning_history/
│   ├── logs/
│   └── permutations_9.txt                 # permutation of goals for train-test-split
├── scripts/ 
│   ├── collect_data.sh                    # run stage 1: dataset collection
│   ├── eval.sh                            # run stage 3: evaluate the trained model
│   └── train_ad.sh                        # run stage 2: train algorithm distillation
├── src/
│   ├── check_in_context/                  # stage 3:
│   │   └── eval_one_model.py              #    evaluate a given model
│   ├── data/
│   │   ├── __init__.py                    # stage 1:
│   │   ├── env.py                         #    env config 
│   │   ├── generate_goals.py              #    goals setup
│   │   └── ppo.py                         #    collect dataset with ppo
│   └── dt                                 # stage 2:
│       ├── eval.py                        #    rollout eval
│       ├── model.py                       #    AD model
│       ├── schedule.py                    #    lr sheduler
│       ├── seq_dataset.py                 #    dataloader 
│       ├── train.py                       #    train script & config
│       └── utils.py
├── static/                                # images for report  
├── runs/                                  # tensorboard dir
└── wandb/                                 # wandb dir  
```


## Quick start 🏃

### 0.0 Setup wandb

To enable [wandb](https://wandb.ai/site) logging you need to create your wandb profile and run the following once:

```text
wandb init
```

To disable wandb logging (for debugging or other reason) you need to run:

```text
wandb disabled
```

To enable wandb logging (when you need to turn on looging again) you need to run:

```text
wandb enabled
```

### 0.1 Generate goals and permuations 

**or** use the provided file [saved_data/permutations_9.txt]().

```python
python src/data/generate_goals.py
```

### 1. Learn PPO agents to collect the dataset

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

Use [notebooks/explore_dataset.ipynb]() to get the statistics about learned trajectories.


### 2. Train AD

**or** load trained models from gdrive via [link](https://drive.google.com/drive/folders/1_pExW9O4SoaraeDZCu05xageE2HBFj_d)

```bash
python src/dt/train.py
```

Also you can run bash script: [scripts/train_ad.sh]()

#### Train AD with reward predictor:

```bash
python src/dt/train.py --config.add-reward_head
```

### 3. Evaluate trained model

Pass the directory with model and yaml config to evaluation script to get evaluation pngs.

```bash
python3 src/check_in_context/eval_one_model.py \
--model-dir /path/to/dir/with/model/and/config
```

## Acknowledgements

The code is based on the following implementations:

 - [In-Context Reinforcement Learning from Noise Distillation](https://github.com/corl-team/ad-eps)
 - [Decision Transformer](https://github.com/corl-team/CORL/blob/main/algorithms/offline/dt.py)
 - [ppo.py from CleanRL](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py)
