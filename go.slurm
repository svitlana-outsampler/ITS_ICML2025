#!/bin/bash

#SBATCH --job-name=qwen_training # Nom de la tâche
#SBATCH --time=05:00:00        # Durée maximale de la tâche (HH:MM:SS)
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Optionnel : Envoie un email à la fin du job
#SBATCH --mail-type=END       # Type d'email (BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=helluy@unistra.fr
#SBATCH --output=train.out
#SBATCH --error=train.err


#cat /proc/cpuinfo
srun rocm-smi
cd /home/u1/helluy/gitlab/llm_ts
source /home/u1/helluy/gitlab/llm_ts/.venv/bin/activate
python ts_train.py --dataset_path test_jsonl.jsonl --model_name Qwen/Qwen2.5-1.5B-Instruct

