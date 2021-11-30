#!/bin/bash
#SBATCH -N 1                              # Total number of CPU nodes requested
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task 8                 # Total number of CPU cores requested
#SBATCH --mem=32G                         # CPU Memory pool for all cores
#SBATCH --partition=gpu  --gres=gpu:3090:1 # Which queue to run on, and what resources to use
                                               # --partition=<queue> - Use the `<queue>` queue
                                               # --gres=gpu:4 - Use 1 GPU of any type
                                               # --gres=gpu:1080ti:1 - Use 1 GTX 1080TI GPU
#SBATCH --time=50:00:00
#SBATCH -o /home/kzl6/slurm/logs/spectral-attack-%j.out                              # Name of stdout output file (%j expands to jobId)
#SBATCH -e /home/kzl6/slurm/logs/spectral-attack-%j.err                             # Name of stderr output file (%j expands to jobId)

# RECIPE="textfooler"
RECIPE="clare"
# MODEL="bert-base-uncased-snli"
# MODEL="bert-base-uncased-ag-news"
MODEL="bert-base-uncased-imdb"
# MODEL="bert-base-uncased-mr"
SAVE_PATH="data/${RECIPE}-${MODEL}.csv"

srun --output=logs/%j.out --error=logs/%j.err --label textattack attack \
    --recipe $RECIPE \
    --model $MODEL \
    --log-to-csv $SAVE_PATH \
    --num-successful-examples 5000 \
    # --num-examples -1
    # # 
