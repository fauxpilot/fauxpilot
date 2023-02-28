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

# On versions above 20.10.2, docker-compose is "docker compose".
DOCKER_STR=$(docker --version)
DOCKER_VER=$(echo "$DOCKER_STR" | sed -n 's/.*\([0-9]\{2\}\.[0-9]\{2\}\.[0-9]\{2\}\).*/\1/p')

echo "$DOCKER_VER ge 20.10.2"
if dpkg --compare-versions $DOCKER_VER gt 20.10.2; then
  DOCKER_CMD="docker-compose"
else
  DOCKER_CMD="docker compose"
fi
$DOCKER_CMD up $options --remove-orphans --build

