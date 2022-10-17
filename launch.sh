#!/usr/bin/env bash

# Read in config.env file; error if not found
if [ ! -f config.env ]; then
    echo "config.env not found, please run setup.sh"
    exit 1
fi
source config.env

export NUM_GPUS=${NUM_GPUS}

# if model name starts with "py-", it means we're dealing with the python backend.
if [[ $(echo "$MODEL" | cut -c1-3) == "py-" ]]; then
    export MODEL_DIR="${MODEL_DIR}"/"${MODEL}" #/py_model"
else
    export MODEL_DIR="${MODEL_DIR}"/"${MODEL}-${NUM_GPUS}gpu"
fi

export GPUS=$(seq 0 $(( NUM_GPUS - 1 )) | paste -sd ',')
export HF_CACHE_DIR=${HF_CACHE_DIR}

# On newer versions, docker-compose is docker compose
if command -v docker-compose > /dev/null; then
    docker compose up
else
    docker-compose up
fi
