#!/usr/bin/env bash
#SBATCH --job-name demo
#SBATCH --ntasks 4
#SBATCH --mem 16G
#SBATCH --partition standard
#SBATCH --gres gpu:1
#SBATCH -D /imatge/alba.herrera/maskMattnet/MAttNet
#SBATCH --output /imatge/alba.herrera/maskMattnet/MAttNet/logs/%x_%u_%j.out

source ../venv/bin/activate
module load cuda/8.0
python cv/demo.py

