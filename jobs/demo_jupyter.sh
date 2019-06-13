#!/usr/bin/env bash

source venv2/bin/activate
module load cuda/8.0

export XDG_RUNTIME_DIR=""
jupyter lab --no-browser --port=7844



