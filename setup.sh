#!/usr/bin/env bash

if [ -f config.env ]; then
    echo "config.env already exists, skipping"
    echo "Please delete config.env if you want to re-run this script"
    exit 0
fi

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
    echo "Converted model for ${MODEL}-${NUM_GPUS}gpu already exists, skipping"
    echo "Please delete ${MODEL_DIR}/${MODEL}-${NUM_GPUS}gpu if you want to re-convert it"
    exit 0
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
