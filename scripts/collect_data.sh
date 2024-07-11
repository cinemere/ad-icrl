#!/bin/bash

for i in {0..81}
do
    # echo "Welcome $i times"
    python3 src/collect_data/collect.py --env.goal-index $i
done