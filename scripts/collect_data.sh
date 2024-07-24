#!/bin/bash

for i in {0..81}
do
    # echo "Welcome $i times"
    # python3 src/collect_data/collect.py --env.goal-index $i
    python3 src/collect_data/ppo.py --env.goal-index $i --track --num-envs 5 --total-timesteps 20000 --learning-rate 0.01
done