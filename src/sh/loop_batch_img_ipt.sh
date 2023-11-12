#!/bin/bash

for min_index in 0 16 32 48 64
  do
    max_index=$((min_index + 16))
        echo loop $min_index $max_index
        sbatch -J ${min_index}_${max_index}_ipt --output=${min_index}_${max_index}_ipt.out pw13_loop_ipt.sh $min_index $max_index
  done

