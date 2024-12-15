#!/bin/sh

#SBATCH -A XXX
#SBATCH -t 24:0:0
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1


module load python/miniconda3.9
module load cuda/11.8
source /share/apps/python/miniconda3.9/etc/profile.d/conda.sh

export PYTHONPATH=~/ETPINN
srun python  cases.py  $1 $2 $3 $4
