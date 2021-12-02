#!/bin/bash
#SBATCH -J 1e-4mlp                # Job name
#SBATCH -o /home/kzl6/slurm/logs/spectral-attack-%j.out                              # Name of stdout output file (%j expands to jobId)
#SBATCH -e /home/kzl6/slurm/logs/spectral-attack-%j.err                             # Name of stderr output file (%j expands to jobId)
#SBATCH -N 1                                 # Total number of nodes requested
#SBATCH -n 4                                 # Total number of cores requested
#SBATCH --mem=20000                          # Total amount of (real) memory requested (per node)
#SBATCH -t 24:00:00                          # Time limit (hh:mm:ss)
#SBATCH --partition=kilian        # Request partition for resource allocation
#SBATCH --gres=gpu:1                        # Specify a list of generic consumable resources (per node)

srun --output=logs/%j.out --error=logs/%j.err --label python train.py \
    --wandb \
    --notes 'mlp lr?' \
    --do_train --do_eval --eval_test \
    --model 'bert-base-uncased' \
    --classifier 'mlp' \
    --attack 'textfooler' \
    --dataset 'snli' \
    --seed 46 \
    --eval_metric f1 \
    --num_train_epochs 10 \
    --eval_per_epoch 4 \
    --output_dir .experiment \
    --scheduler constant \
    --train_batch_size 40 \
    --eval_batch_size 10 \
    --gradient_accumulation_steps 4  \
    --learning_rate 1e-4