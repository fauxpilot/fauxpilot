#!/usr/bin/env bash

# Read in .env file; error if not found
if [ ! -f .env ]; then
    echo ".env not found, running setup.sh"
    bash setup.sh
fi
source .env

export NUM_GPUS=${NUM_GPUS}
export GPUS=$(seq 0 $(( NUM_GPUS - 1 )) | paste -sd ',')

# if model name starts with "py-", it means we're dealing with the python backend.
if [[ $(echo "$MODEL" | cut -c1-3) == "py-" ]]; then
    export MODEL_DIR="${MODEL_DIR}"/"${MODEL}" #/py_model"
else
    export MODEL_DIR="${MODEL_DIR}"/"${MODEL}-${NUM_GPUS}gpu"
fi

export HF_CACHE_DIR=${HF_CACHE_DIR}

# On newer versions, docker-compose is docker compose
docker compose up -d --remove-orphans || docker-compose up -d --remove-orphans
