#!/usr/bin/env bash

# Read in .env file; error if not found
if [ ! -f .env ]; then
    echo ".env not found, running setup.sh"
    bash setup.sh
fi
source .env

function showhelp () {
   # Display Help
   echo
   echo "Usage: $0 [option...]"
   echo "options:"
   echo "  -h       Print this help."
   echo "  -d       Start in daemon mode."
   echo
}

while getopts "hd" option; do
   case $option in
      h)
         showhelp
         exit;;
      d)
         options="-d"
         ;;
     \?) # incorrect option
         echo "Error: Invalid option"
         exit;;
   esac
done

# On versions above 20.10.2, docker-compose is docker compose
smaller=$(printf "$(docker --version | egrep -o '[0-9]+\.[0-9]+\.[0-9]+')\n20.10.2" | sort -V | head -n1)
if [[ "$smaller" == "20.10.2" ]]; then
  docker compose up $options --remove-orphans --build
else
  docker-compose up $options --remove-orphans --build
fi;
