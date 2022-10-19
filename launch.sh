#!/usr/bin/env bash

# Read in .env file; error if not found
if [ ! -f .env ]; then
    echo ".env not found, running setup.sh"
    bash setup.sh
fi
source .env

# On newer versions, docker-compose is docker compose
docker compose up -d --remove-orphans || docker-compose up -d --remove-orphans
