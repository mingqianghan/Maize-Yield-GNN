#!/bin/bash
# Windows Aanconda

# Activate Anaconda environment (local)
# source /d/Software/Anaconda/etc/profile.d/conda.sh
# conda activate base


for model in SVR RF GBR ensemble
do
    for tp in R1 R2 R3 R4 R5 R6
    do
        for seed in 0 1 2 
        do 
            python main.py --method $model --timepoint $tp --seed $seed
        done
    done
done

