# ad-icrl
Implementation of paper "In-context Reinforcement Learning with Algorithm Distillation"

## Installation 🧑‍🔧

```bash
git clone ...
cd ad-icrl
python -m venv venv
pip install -r requirements.txt
```

## Quick start 🏃

1. **Generate goals and permuations**

```python
python src/collect_data/generate_goals.py
```

2. **Learn A2C agents to collect the dataset**

```python
chmod +x ./scripts/collect_data.sh
./scripts/collect_data.sh
```