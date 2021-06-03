#! /bin/sh

./FlexibleNN \
--name testest \
--max_episodes 500000 \
--seed 5 \
--steps 40000000 \
--width 0 \
--step_size 3e-5 \
--num_layers 0 \
--add_features true \
--features_min_timesteps 50000 \
--num_new_features 1 \
--gamma 0.98  \
--lambda 0.99 \
--epsilon 0.05 \
--episode_length 1000 \
--episode_gap 5 \
--tmaze_corridor_length 2 \
--prediction_problem true \
--comment $1
