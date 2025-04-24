#!/bin/bash

#SBATCH --job-name=qwen_training # Nom de la tâche
#SBATCH --time=05:00:00        # Durée maximale de la tâche (HH:MM:SS)
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Optionnel : Envoie un email à la fin du job
#SBATCH --mail-type=END       # Type d'email (BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=helluy@unistra.fr

cat /proc/cpuinfo
srun rocm-smi
cd /home/u1/helluy/gitlab/llm_ts
srun source /home/u1/helluy/gitlab/llm_ts/.venv/bin/activate
srun python train.py

