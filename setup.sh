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

############### Common configuration ###############

# Read number of GPUs
read -rp "Enter number of GPUs [1]: " NUM_GPUS
NUM_GPUS=${NUM_GPUS:-1}

read -rp "External port for the API [5000]: " API_EXTERNAL_PORT
API_EXTERNAL_PORT=${API_EXTERNAL_PORT:-5000}

read -rp "Address for Triton [triton]: " TRITON_HOST
TRITON_HOST=${TRITON_HOST:-triton}

read -rp "Port of Triton host [8001]: " TRITON_PORT
TRITON_PORT=${TRITON_PORT:-8001}

# Read models root directory (all models go under this)
read -rp "Where do you want to save your models [$(pwd)/models]? " MODELS_ROOT_DIR
if [ -z "$MODELS_ROOT_DIR" ]; then
    MODELS_ROOT_DIR="$(pwd)/models"
else
    MODELS_ROOT_DIR="$(readlink -m "${MODELS_ROOT_DIR}")"
fi
mkdir -p "$MODELS_ROOT_DIR"

# Write .env
echo "NUM_GPUS=${NUM_GPUS}" >> .env
echo "GPUS=$(seq 0 $(( NUM_GPUS - 1)) | paste -s -d ',' -)" >> .env
echo "API_EXTERNAL_PORT=${API_EXTERNAL_PORT}" >> .env
echo "TRITON_HOST=${TRITON_HOST}" >> .env
echo "TRITON_PORT=${TRITON_PORT}" >> .env

############### Backend specific configuration ###############

function fastertransformer_backend(){
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

    echo "MODEL=${MODEL}" >> .env
    echo "MODEL_DIR=${MODELS_ROOT_DIR}/${MODEL}-${NUM_GPUS}gpu" >> .env

    if (test -d "$MODELS_ROOT_DIR"/"${MODEL}"-"${NUM_GPUS}"gpu ); then
      echo "$MODELS_ROOT_DIR"/"${MODEL}"-"${NUM_GPUS}"gpu
      echo "Converted model for ${MODEL}-${NUM_GPUS}gpu already exists."
      read -rp "Do you want to re-use it? y/n: " REUSE_CHOICE
      if [[ ${REUSE_CHOICE:-y} =~ ^[Yy]$ ]]
      then
        DOWNLOAD_MODEL=n
        echo "Re-using model"
      else
        DOWNLOAD_MODEL=y
        rm -rf "$MODELS_ROOT_DIR"/"${MODEL}"-"${NUM_GPUS}"gpu
      fi
    else
      DOWNLOAD_MODEL=y
    fi

    if [[ ${DOWNLOAD_MODEL:-y} =~ ^[Yy]$ ]]
    then
      if [ "$NUM_GPUS" -le 2 ]; then
        echo "Downloading the model from HuggingFace, this will take a while..."
        SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
        DEST="${MODEL}-${NUM_GPUS}gpu"
        ARCHIVE="${MODELS_ROOT_DIR}/${DEST}.tar.zst"
        cp -r "$SCRIPT_DIR"/converter/models/"$DEST" "${MODELS_ROOT_DIR}"
        curl -L "https://huggingface.co/moyix/${MODEL}-gptj/resolve/main/${MODEL}-${NUM_GPUS}gpu.tar.zst" \
            -o "$ARCHIVE"
        zstd -dc "$ARCHIVE" | tar -xf - -C "${MODELS_ROOT_DIR}"
        rm -f "$ARCHIVE"
      else
        echo "Downloading and converting the model, this will take a while..."
        docker run --rm -v "${MODELS_ROOT_DIR}":/models -e MODEL=${MODEL} -e NUM_GPUS="${NUM_GPUS}" moyix/model_converter:latest
      fi
    fi

    # Not used for this backend but needs to be present
    HF_CACHE_DIR="$(pwd)/.hf_cache"
    mkdir -p "$HF_CACHE_DIR"
    echo "HF_CACHE_DIR=${HF_CACHE_DIR}" >> .env
}

function python_backend(){
    echo "Models available:"
    echo "[1] codegen-350M-mono (1GB total VRAM required; Python-only)"
    echo "[2] codegen-350M-multi (1GB total VRAM required; multi-language)"
    echo "[3] codegen-2B-mono (4GB total VRAM required; Python-only)"
    echo "[4] codegen-2B-multi (4GB total VRAM required; multi-language)"

    read -rp "Enter your choice [4]: " MODEL_NUM

    # Convert model number to model name
    case $MODEL_NUM in
        1) MODEL="codegen-350M-mono"; ORG="Salesforce" ;;
        2) MODEL="codegen-350M-multi"; ORG="Salesforce" ;;
        3) MODEL="codegen-2B-mono"; ORG="Salesforce" ;;
        4) MODEL="codegen-2B-multi"; ORG="Salesforce" ;;
        *) MODEL="codegen-2B-multi"; ORG="Salesforce" ;;
    esac

    # share huggingface cache? Should be safe to share, but permission issues may arise depending upon your docker setup
    read -rp "Do you want to share your huggingface cache between host and docker container? y/n [n]: " SHARE_HF_CACHE
    SHARE_HF_CACHE=${SHARE_HF_CACHE:-n}
    if [[ ${SHARE_HF_CACHE:-y} =~ ^[Yy]$ ]]; then
        read -rp "Enter your huggingface cache directory [$HOME/.cache/huggingface]: " HF_CACHE_DIR
        HF_CACHE_DIR=${HF_CACHE_DIR:-$HOME/.cache/huggingface}
    else
        HF_CACHE_DIR="$(pwd)/.hf_cache"
    fi

    # use int8? Allows larger models to fit in GPU but might be very marginally slower
    read -rp "Do you want to use int8? y/n [n]: " USE_INT8
    if [[ ! $USE_INT8 =~ ^[Yy]$ ]]; then
        USE_INT8="0"
    else
        USE_INT8="1"
    fi

    # Write config.env
    echo "MODEL=py-${MODEL}" >> .env
    echo "MODEL_DIR=${MODELS_ROOT_DIR}/py-${ORG}-${MODEL}" >> .env  # different format from fastertransformer backend
    echo "HF_CACHE_DIR=${HF_CACHE_DIR}" >> .env

    python3 ./python_backend/init_model.py --model_name "${MODEL}" --org_name "${ORG}" --model_dir "${MODELS_ROOT_DIR}" --use_int8 "${USE_INT8}"
    bash -c "source .env ; docker compose build || docker-compose build"
}

# choose backend
echo "Choose your backend:"
echo "[1] FasterTransformer backend (faster, but limited models)"
echo "[2] Python backend (slower, but more models, and allows loading with int8)"
read -rp "Enter your choice [1]: " BACKEND_NUM

if [[ "$BACKEND_NUM" -eq 2 ]]; then
    python_backend
else
    fastertransformer_backend
fi

read -rp "Config complete, do you want to run FauxPilot? [y/n] " RUN
if [[ ${RUN:-y} =~ ^[Yy]$ ]]
then
  bash ./launch.sh
else
  echo "You can run ./launch.sh to start the FauxPilot server."
  exit 0
fi
