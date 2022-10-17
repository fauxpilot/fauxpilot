#!/usr/bin/env bash

if [ -f config.env ]; then
    echo "config.env already exists, skipping"
    echo "Please delete config.env if you want to re-run this script"
    exit 1
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
    read -p "Enter your choice [6]: " MODEL_NUM

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
    read -p "Enter number of GPUs [1]: " NUM_GPUS
    NUM_GPUS=${NUM_GPUS:-1}

    # Read model directory
    read -p "Where do you want to save the model [$(pwd)/models]? " MODEL_DIR
    if [ -z "$MODEL_DIR" ]; then
        MODEL_DIR="$(pwd)/models"
    else
        MODEL_DIR="$(readlink -m "${MODEL_DIR}")"
    fi

    # Write config.env
    echo "MODEL=${MODEL}" > config.env
    echo "NUM_GPUS=${NUM_GPUS}" >> config.env
    echo "MODEL_DIR=${MODEL_DIR}" >> config.env

    if [ -d "$MODEL_DIR"/"${MODEL}"-${NUM_GPUS}gpu ]; then
        echo "Converted model for ${MODEL}-${NUM_GPUS}gpu already exists."
        read -p "Do you want to re-use it? y/n: " REUSE_CHOICE
        if [ "${REUSE_CHOICE^^}" = "Y" ]; then
            exit 0
        fi
    fi

    # Create model directory
    mkdir -p "${MODEL_DIR}"

    # For some of the models we can download it preconverted.
    if [ $NUM_GPUS -le 2 ]; then
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
        docker run --rm -v ${MODEL_DIR}:/models -e MODEL=${MODEL} -e NUM_GPUS=${NUM_GPUS} moyix/model_converter:latest
    fi
    echo "Done! Now run ./launch.sh to start the FauxPilot server."
}

function python_backend(){
    echo "Models available:"
    echo "[1] codegen-350M-mono (1GB total VRAM required; Python-only)"
    echo "[2] codegen-350M-multi (1GB total VRAM required; multi-language)"
    echo "[3] codegen-2B-mono (4GB total VRAM required; Python-only)"
    echo "[4] codegen-2B-multi (4GB total VRAM required; multi-language)"
    # echo "[5] codegen-6B-mono (13GB total VRAM required; Python-only)"
    # echo "[6] codegen-6B-multi (13GB total VRAM required; multi-language)"
    # echo "[7] codegen-16B-mono (32GB total VRAM required; Python-only)"
    # echo "[8] codegen-16B-multi (32GB total VRAM required; multi-language)"
    # Read their choice
    read -p "Enter your choice [4]: " MODEL_NUM

    # Convert model number to model name
    case $MODEL_NUM in
        1) MODEL="codegen-350M-mono"; ORG="Salesforce" ;;
        2) MODEL="codegen-350M-multi"; ORG="Salesforce" ;;
        3) MODEL="codegen-2B-mono"; ORG="Salesforce" ;;
        4) MODEL="codegen-2B-multi"; ORG="Salesforce" ;;
    esac

    # Read number of GPUs -- not strictly required for python backend, because of device_map="auto",
    # but docker-compose.py uses it to select CUDA_VISIBLE_DEVICES
    read -p "Enter number of GPUs [1]: " NUM_GPUS
    NUM_GPUS=${NUM_GPUS:-1}

    # Read model directory
    read -p "Where do you want to save the model [$(pwd)/models]? " MODEL_DIR
    MODEL_DIR=${MODEL_DIR:-$(pwd)/models}
    if [ -z "$MODEL_DIR" ]; then
        MODEL_DIR="$(pwd)/models"
    else
        MODEL_DIR="$(readlink -m "${MODEL_DIR}")"
    fi

    # share huggingface cache? Should be safe to share, but permission issues may arise depending upon your docker setup
    read -p "Do you want to share your huggingface cache between host and docker container? y/n [n]: " SHARE_HF_CACHE
    SHARE_HF_CACHE=${SHARE_HF_CACHE:-n}
    if [ "${SHARE_HF_CACHE^^}" = "Y" ]; then
        read -p "Enter your huggingface cache directory [$HOME/.cache/huggingface]: " HF_CACHE_DIR
        HF_CACHE_DIR=${HF_CACHE_DIR:-$HOME/.cache/huggingface}
    else
        HF_CACHE_DIR="/tmp/hf_cache"
    fi

    # use int8? Allows larger models to fit in GPU but might be very marginally slower
    read -p "Do you want to use int8? y/n [y]: " USE_INT8
    USE_INT8=${USE_INT8:-y}
    if [ "${USE_INT8^^}" = "N" ]; then
        USE_INT8="0"
    else
        USE_INT8="1"
    fi

    # Write config.env
    echo "MODEL=py-${MODEL}" > config.env
    echo "NUM_GPUS=${NUM_GPUS}" >> config.env
    echo "MODEL_DIR=${MODEL_DIR}" >> config.env
    echo "HF_CACHE_DIR=${HF_CACHE_DIR}" >> config.env

    # Create model directory
    mkdir -p "${MODEL_DIR}/"
    python3 ./python_backend/init_model.py --model_name "${MODEL}" --org_name "${ORG}" --model_dir "${MODEL_DIR}" --use_int8 "${USE_INT8}"
    
    echo "Done! Now run ./launch.sh to start the FauxPilot server."
}

# choose backend
echo "Choose your backend:"
echo "[1] FasterTransformer backend (faster, but limited models)"
echo "[2] Python backend (slower, but more models, and allows loading with int8)"
read -p "Enter your choice [1]: " BACKEND_NUM

if [ $BACKEND_NUM -eq 2 ]; then
    python_backend
else
    fastertransformer_backend
fi