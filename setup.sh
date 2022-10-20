#!/usr/bin/env bash

if [ -f .env ]; then
    read -rp ".env already exists, do you want to delete .env and recreate it? [y/n] " DELETE
    if [[ ${DELETE:-y} =~ ^[Yy]$ ]]
    then
      echo "Deleting .env"
      rm .env
    else
      echo "Exiting"
      exit 0
    fi;
fi

function check_dep(){
    echo "Checking for $1 ..."
    which "$1" 2>/dev/null || {
        echo "Please install $1."
        exit 1
    }
}
check_dep curl
check_dep zstd
check_dep docker


echo "Models available:"
echo "[1] codegen-350M-mono (2GB total VRAM required; Python-only)"
echo "[2] codegen-350M-multi (2GB total VRAM required; multi-language)"
echo "[3] codegen-2B-mono (7GB total VRAM required; Python-only)"
echo "[4] codegen-2B-multi (7GB total VRAM required; multi-language)"
echo "[5] codegen-6B-mono (13GB total VRAM required; Python-only)"
echo "[6] codegen-6B-multi (13GB total VRAM required; multi-language)"
echo "[7] codegen-16B-mono (32GB total VRAM required; Python-only)"
echo "[8] codegen-16B-multi (32GB total VRAM required; multi-language)"
# Read their choice
read -rp "Enter your choice [6]: " MODEL_NUM

# Convert model number to model name
case $MODEL_NUM in
    1) MODEL="codegen-350M-mono" ;;
    2) MODEL="codegen-350M-multi" ;;
    3) MODEL="codegen-2B-mono" ;;
    4) MODEL="codegen-2B-multi" ;;
    5) MODEL="codegen-6B-mono" ;;
    6) MODEL="codegen-6B-multi" ;;
    7) MODEL="codegen-16B-mono" ;;
    8) MODEL="codegen-16B-multi" ;;
    *) MODEL="codegen-6B-multi" ;;
esac

# Read number of GPUs
read -rp "Enter number of GPUs [1]: " NUM_GPUS
NUM_GPUS=${NUM_GPUS:-1}

read -rp "External port for the API [5000]: " API_EXTERNAL_PORT
API_EXTERNAL_PORT=${API_EXTERNAL_PORT:-5000}

read -rp "Address for Triton [triton]: " TRITON_HOST
TRITON_HOST=${TRITON_HOST:-triton}

read -rp "Port of Triton host [8001]: " TRITON_PORT
TRITON_PORT=${TRITON_PORT:-8001}

# Read model directory
read -rp "Where do you want to save the model [$(pwd)/models]? " MODEL_DIR
if [ -z "$MODEL_DIR" ]; then
    MODEL_DIR="$(pwd)/models"
else
    MODEL_DIR="$(readlink -m "${MODEL_DIR}")"
fi

# Write .env
echo "MODEL=${MODEL}" > .env
echo "NUM_GPUS=${NUM_GPUS}" >> .env
echo "MODEL_DIR=${MODEL_DIR}/${MODEL}-${NUM_GPUS}gpu" >> .env
echo "API_EXTERNAL_PORT=${API_EXTERNAL_PORT}" >> .env
echo "TRITON_HOST=${TRITON_HOST}" >> .env
echo "TRITON_PORT=${TRITON_PORT}" >> .env
echo "GPUS=$(seq 0 $(( NUM_GPUS - 1)) | paste -s -d ',' -)" >> .env

if (test -d "$MODEL_DIR"/"${MODEL}"-"${NUM_GPUS}"gpu ); then
  echo "$MODEL_DIR"/"${MODEL}"-"${NUM_GPUS}"gpu
  echo "Converted model for ${MODEL}-${NUM_GPUS}gpu already exists."
  read -rp "Do you want to re-use it? y/n: " REUSE_CHOICE
  if [[ ${REUSE_CHOICE:-y} =~ ^[Yy]$ ]]
  then
    DOWNLOAD_MODEL=n
    echo "Re-using model"
  else
    DOWNLOAD_MODEL=y
    rm -rf "$MODEL_DIR"/"${MODEL}"-"${NUM_GPUS}"gpu
  fi
else
  DOWNLOAD_MODEL=y
fi

if [[ ${DOWNLOAD_MODEL:-y} =~ ^[Yy]$ ]]
then
  # Create model directory
  mkdir -p "${MODEL_DIR}"
  if [ "$NUM_GPUS" -le 2 ]; then
    echo "Downloading the model from HuggingFace, this will take a while..."
    SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
    DEST="${MODEL}-${NUM_GPUS}gpu"
    ARCHIVE="${MODEL_DIR}/${DEST}.tar.zst"
    cp -r "$SCRIPT_DIR"/converter/models/"$DEST" "${MODEL_DIR}"
    curl -L "https://huggingface.co/moyix/${MODEL}-gptj/resolve/main/${MODEL}-${NUM_GPUS}gpu.tar.zst" \
        -o "$ARCHIVE"
    zstd -dc "$ARCHIVE" | tar -xf - -C "${MODEL_DIR}"
    rm -f "$ARCHIVE"
  else
    echo "Downloading and converting the model, this will take a while..."
    docker run --rm -v "${MODEL_DIR}":/models -e MODEL=${MODEL} -e NUM_GPUS="${NUM_GPUS}" moyix/model_converter:latest
  fi
fi

MODEL_DIR="${MODEL_DIR}"/"${MODEL}-${NUM_GPUS}gpu"
CONF_PATH="/model/fastertransformer/config.pbtxt"
GPUS=$(seq 0 $(( NUM_GPUS - 1 )) | paste -sd ',')

docker run --rm -v ${MODEL_DIR}:/model --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=${GPUS} txy:latest /bin/bash "python3 tune && cp ./config.ini /model/config.ini"

read -rp "Config complete, do you want to run FauxPilot? [y/n] " RUN
if [[ ${RUN:-y} =~ ^[Yy]$ ]]
then
  bash ./launch.sh
else
  echo "You can run ./launch.sh to start the FauxPilot server."
  exit 0
fi;
