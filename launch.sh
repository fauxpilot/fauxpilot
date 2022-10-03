#!/usr/bin/env bash

# Read in .env file; error if not found
if [ ! -f .env ]; then
    echo ".env not found, running setup.sh"
    bash setup.sh
fi
source .env

# On newer versions, docker-compose is docker compose
DOCKER_COMPOSE=$(command -v docker-compose)
if [ -z "$DOCKER_COMPOSE" ]; then
    DOCKER_COMPOSE="docker compose"
fi

$DOCKER_COMPOSE up
