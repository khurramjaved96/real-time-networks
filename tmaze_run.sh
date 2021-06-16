#! /bin/sh

./FlexibleNN \
--name testest1 \
--max_episodes 500000 \
--seed 4 \
--steps 40000000 \
--width 0 \
--step_size 3e-5 \
--num_layers 0 \
--add_features true \
--features_min_timesteps 80000 \
--num_new_features 10 \
--gamma 0.98  \
--lambda 0.98 \
--epsilon 0.05 \
--episode_length 1000 \
--episode_gap 8 \
--tmaze_corridor_length 2 \
--prediction_problem false \
--comment $1
