#!/usr/bin/env bash

# Read in config.env file; error if not found
if [ ! -f config.env ]; then
    echo "config.env not found, please run setup.sh"
    exit 1
fi
source config.env
source copilot_proxy.env

# On newer versions, docker-compose is docker compose
DOCKER_COMPOSE=$(command -v docker-compose)
if [ -z "$DOCKER_COMPOSE" ]; then
    DOCKER_COMPOSE="docker compose"
fi

# services triton
export NUM_GPUS=${NUM_GPUS}
export MODEL_DIR="${MODEL_DIR}"/"${MODEL}-${NUM_GPUS}gpu"
export GPUS=$(seq 0 $(( NUM_GPUS - 1 )) | paste -sd ',')
# services copilot_proxy
export API_HOST=${API_HOST}
export API_PORT=${API_PORT}

export
$DOCKER_COMPOSE up
