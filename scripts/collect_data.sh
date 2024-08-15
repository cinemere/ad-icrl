#!/bin/bash
export PYTHONPATH=.

for i in {0..80}
do
    # python3 src/collect_data/collect.py --env.goal-index $i
    # python3 src/collect_data/ppo.py --env.goal-index $i --track --num-envs 5 --total-timesteps 20000 --learning-rate 0.01
    # python3 src/collect_data/ppo.py --env.goal-index $i --num-envs 10 --total-timesteps 20000 --learning-rate 0.001 --num-steps 10 --num-minibatches 1 --track
    # python3 src/collect_data/ppo.py --env.goal-index $i --num-envs 100 --total-timesteps 200000 --learning-rate 0.0001 --num-steps 1 --num-minibatches 1 --track
    python3 src/collect_data/ppo.py --env.goal-index $i --num-envs 100 --total-timesteps 200000 --learning-rate 0.0002 --num-steps 1 --num-minibatches 1 --track --seed 2
done