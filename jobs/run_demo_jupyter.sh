#!/usr/bin/env bash

srun --mem=16G --cpus-per-task=4 --gres=gpu:1 --partition=standard \
     --job-name=notebook -D /imatge/alba.herrera/maskMattnet/MAttNet \
     --tunnel 7833:7833 ./jobs/demo_jupyter.sh --port 7833
