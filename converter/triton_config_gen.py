#!/usr/bin/env python

import argparse
import os
from string import Template
from transformers import GPTJConfig, AutoTokenizer
import torch


def round_up(x, multiple):
    remainder = x % multiple
    return x if remainder == 0 else x + multiple - remainder


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
CONFIG_TEMPLATE_PATH = os.path.join(SCRIPT_DIR, 'config_template.pbtxt')

# Generate a config file for a CodeGen model for use with Triton

parser = argparse.ArgumentParser('Create Triton config files for CodeGen models')
parser.add_argument('--template', default=CONFIG_TEMPLATE_PATH, help='Path to the config template')
parser.add_argument('--model_store', required=True, help='Path to the Triton model store')
parser.add_argument('--hf_model_dir', required=True, help='Path to HF model directory')
parser.add_argument('--tokenizer', default='Salesforce/codegen-16B-multi', help='Name or path to the tokenizer')
parser.add_argument('--rebase', default=None, help='Path to rebase the model store to (e.g. for Docker)')
parser.add_argument('-n', '--num_gpu', help='Number of GPUs to use', type=int, default=1)
args = parser.parse_args()

# Vars we need to fill in:
# name
# tensor_para_size
# max_seq_len
# is_half
# head_num
# size_per_head
# inter_size
# vocab_size
# start_id
# end_id
# decoder_layers
# name
# rotary_embedding
# checkpoint_path

# Global options
if args.hf_model_dir.endswith('/'):
    args.hf_model_dir = args.hf_model_dir[:-1]
config = GPTJConfig.from_pretrained(args.hf_model_dir)
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
max_seq_len = config.n_positions
is_half = '1' if config.torch_dtype == torch.float16 else '0'

# Read in the template config file
with open(args.template, 'r') as f:
    template = Template(f.read())

model_name = os.path.basename(args.hf_model_dir)
version = '1'
params = {}
params['tensor_para_size'] = args.num_gpu
params['name'] = model_name
params['max_seq_len'] = max_seq_len
params['is_half'] = is_half
params['head_num'] = config.n_head
params['size_per_head'] = config.n_embd // config.n_head
params['inter_size'] = 4*config.n_embd
# Vocab size *sometimes* gets rounded up to a multiple of 1024
params['vocab_size'] = tokenizer.vocab_size+len(tokenizer.get_added_vocab())  # round_up(tokenizer.vocab_size, 1024)
params['start_id'] = tokenizer.eos_token_id
params['end_id'] = tokenizer.eos_token_id
params['decoder_layers'] = config.n_layer
params['rotary_embedding'] = config.rotary_dim
# NOTE: this assumes that the model dir follows the format used by the other conversion scripts
model_dir = os.path.join(args.model_store, f'{model_name}-{args.num_gpu}gpu')
weights_path = os.path.join(model_dir, 'fastertransformer', f'{version}', f'{args.num_gpu}-gpu')
if args.rebase:
    rebased_model_dir = os.path.join(args.rebase, f'{model_name}-{args.num_gpu}gpu')
    rebased_weights_path = os.path.join(args.rebase, 'fastertransformer', f'{version}', f'{args.num_gpu}-gpu')
else:
    rebased_model_dir = model_dir
    rebased_weights_path = weights_path

params['checkpoint_path'] = rebased_weights_path
triton_config = template.substitute(params)
assert '${' not in triton_config

# Make directory structure
os.makedirs(weights_path, exist_ok=True)

# Write config file
config_path = os.path.join(model_dir, 'fastertransformer', 'config.pbtxt')
with open(config_path, 'w') as f:
    f.write(triton_config)

print('==========================================================')
print(f'Created config file for {model_name}')
print(f'  Config:      {config_path}')
print(f'  Weights:     {weights_path}')
print(f'  Store:       {args.model_store}')
print(f'  Rebase:      {model_dir} => {args.rebase}')
print(f'    Weights:   {rebased_weights_path}')
print(f'  Num GPU:     {args.num_gpu}')
print('==========================================================')
