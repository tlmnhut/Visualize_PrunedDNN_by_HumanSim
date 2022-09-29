#!/bin/bash

for brain_area in "vTC" "PPA"
do
    for filter_size in {24..56..4}
    do
        for use_pruning in 0 1
        do
            echo "$brain_area $filter_size $use_pruning"
            python compute_perturbation.py --brain_area $brain_area --filter_size $filter_size --use_pruning $use_pruning
        done
    done
done
