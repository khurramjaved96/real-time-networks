#! /bin/sh

./FlexibleNN \
--name cpy_test1 \
--seed 0 \
--steps 10000000 \
--width 10 \
--step_size 3e-4 \
--num_layers 10 \
--data_driven_initialization false \
--fix_L_val 0 \
--randomize_sequence_length false \
--add_features true \
--features_min_timesteps 100000 \
--features_acc_thresh 0.90 \
--features_no_min_L false \
--num_new_features 5 \
--sparsity 0 \
--sequence_gap 1 \
--comment $1
