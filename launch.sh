#!/usr/bin/env bash

# Read in .env file; error if not found
if [ ! -f .env ]; then
    echo ".env not found, running setup.sh"
    bash setup.sh
fi
source .env

# On newer versions, docker-compose is docker compose
if command -v docker-compose > /dev/null; then
    docker compose up
else
    docker-compose up
fi
