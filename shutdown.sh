#!/usr/bin/env bash

source .env

# On newer versions, docker-compose is docker compose
docker compose down --remove-orphans || docker-compose down --remove-orphans
