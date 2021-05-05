#! /bin/sh

./FlexibleNN \
--name cpy_env1 \
--seed 0 \
--steps 10000000 \
--width 500 \
--step_size 3e-5 \
--num_layers 15 \
--data_driven_initialization false \
--randomize_sequence_length false \
--add_features false \
--sparsity 98 \
--sequence_gap 5 \
--comment $1
