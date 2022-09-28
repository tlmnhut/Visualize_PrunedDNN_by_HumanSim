#!/bin/bash

for brain_area in "vTC" "PPA" "FFA"
do
    for filter_size in 0
    do
        for use_pruning in 0 1
        do
            echo "$brain_area $filter_size $use_pruning"
            python saliency_map.py --brain_area $brain_area --filter_size $filter_size --use_pruning $use_pruning
        done
    done
done
