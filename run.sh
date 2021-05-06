#! /bin/sh

./FlexibleNN \
--name cpy_env3 \
--seed 0 \
--steps 5000000 \
--width 20 \
--step_size 3e-4 \
--num_layers 10 \
--data_driven_initialization false \
--fix_L_val 0 \
--randomize_sequence_length false \
--add_features false \
--sparsity 0 \
--sequence_gap 1 \
--comment $1
