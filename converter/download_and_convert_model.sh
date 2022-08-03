#!/bin/bash

MODEL=${1}
NUM_GPUS=${2}

echo "Converting model ${MODEL} with ${NUM_GPUS} GPUs"

cp -r models/${MODEL}-${NUM_GPUS}gpu /models
python3 codegen_gptj_convert.py --code_model Salesforce/${MODEL} ${MODEL}-hf
python3 huggingface_gptj_convert.py -in_file ${MODEL}-hf -saved_dir /models/${MODEL}-${NUM_GPUS}gpu/fastertransformer/1 -infer_gpu_num ${NUM_GPUS}
rm -rf ${MODEL}-hf
