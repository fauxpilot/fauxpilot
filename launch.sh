#!/usr/bin/env bash

# Read in .env file; error if not found
if [ ! -f .env ]; then
    echo ".env not found, running setup.sh"
    bash setup.sh
fi
source .env

function help() {
   # Display Help
   echo
   echo "Usage: $0 [option...]"
   echo "options:"
   echo "-h, --help       Print this Help."
   echo "-d, --daemon     Start with a daemon mode."
   echo
}

while getopts ":h:d" option; do
   case $option in
      h)
         help
         exit;;
      d)
         options="-d"
         echo "-d option"
         ;;
     \?) # incorrect option
         echo "Error: Invalid option"
         exit;;
   esac
done

# On newer versions, docker-compose is docker compose
docker compose up $options --remove-orphans || docker-compose up $options --remove-orphans
