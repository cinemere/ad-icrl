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

## Quick start 🏃

1. **Generate goals and permuations**

```python
python src/collect_data/generate_goals.py
```

2. **Learn A2C agents to collect the dataset**

Run the script to create the trajectories:

```python
chmod +x ./scripts/collect_data.sh
./scripts/collect_data.sh
```

Observe the learning process on tensorboard:
```bash
tensorboard --logdir saved_data/logs/
```

**or** load the trajectories from gdrive via TODO.

3. **Train AD**

turn on wandb

```bash
python src/dt/train.py
```