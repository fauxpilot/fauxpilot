#!/usr/bin/env python

import argparse
from distutils.version import StrictVersion
from google.protobuf import text_format
from model_config_pb2 import ModelConfig, TYPE_UINT64, TYPE_UINT32, TYPE_FP32
from configparser import ConfigParser
import sys, os

# Upgrade model config file from v1.1 to v1.3

parser = argparse.ArgumentParser('Upgrade model config file from v1.1 to v1.3')
parser.add_argument('model_dir', help='Path to the input model')
args = parser.parse_args()

# Make this an absolute path
model_dir = os.path.realpath(args.model_dir)

# Path to the protobuf-text config file
config_path = os.path.join(model_dir, 'fastertransformer', 'config.pbtxt')

# Check for existing backup files and bail if so
old_version = '1.1'
new_version = '1.3'
# Check for a .version file. If it exists, we've already upgraded this model.
# This will also let us use the .version file for future upgrades.
version_path = os.path.join(model_dir, 'fastertransformer', '.version')
if os.path.exists(version_path):
    with open(version_path, 'r') as f:
        old_version = f.read().strip()
    if StrictVersion(old_version) >= StrictVersion(new_version):
        print(f'INFO: model already upgraded to version {old_version}; nothing to do',
            file=sys.stderr)
        sys.exit(0)

backup_ext = f'.bk_{new_version}'
if os.path.exists(config_path+backup_ext):
    print(f'INFO: backup {config_path+backup_ext} already exists; did you already run this script?',
        file=sys.stderr)
    sys.exit(1)

# Read the old config
with open(config_path, 'r') as f:
    config = ModelConfig()
    text_format.Parse(f.read(), config)

# Only support GPT-J for now; we don't have any other model types
if config.parameters['model_type'].string_value != 'GPT-J':
    print(f'ERROR: only GPT-J models are supported for now', file=sys.stderr)
    sys.exit(1)

# Build up the new config
new_config = ModelConfig()
new_config.name = config.name
new_config.backend = config.backend
new_config.default_model_filename = config.default_model_filename
new_config.max_batch_size = config.max_batch_size

# New: model_transaction_policy controls whether to stream tokens. Default
# to false to preserve current behavior.
new_config.model_transaction_policy.decoupled = False

# Inputs
common_inputs = set([
    'input_ids', 'start_id', 'end_id', 'input_lengths', 'request_output_len',
    'runtime_top_k', 'runtime_top_p', 'beam_search_diversity_rate', 'temperature',
    'len_penalty', 'repetition_penalty', 'random_seed', 'is_return_log_probs',
    'beam_width', 'bad_words_list', 'stop_words_list'
])
for input in config.input:
    if input.name not in common_inputs: continue
    new_input = new_config.input.add()
    new_input.CopyFrom(input)
    # Random seed dtype changed from int32 to uint64
    if input.name == 'random_seed':
        new_input.data_type = TYPE_UINT64

# New inputs
# {
#   name: "prompt_learning_task_name_ids"
#   data_type: TYPE_UINT32
#   dims: [ 1 ]
#   reshape: { shape: [ ] }
#   optional: true
# }
new_input = new_config.input.add()
new_input.name = 'prompt_learning_task_name_ids'
new_input.data_type = TYPE_UINT32
new_input.dims.extend([1])
new_input.reshape.shape.extend([])
new_input.optional = True
# {
#   name: "top_p_decay"
#   data_type: TYPE_FP32
#   dims: [ 1 ]
#   reshape: { shape: [ ] }
#   optional: true
# }
new_input = new_config.input.add()
new_input.name = 'top_p_decay'
new_input.data_type = TYPE_FP32
new_input.dims.extend([1])
new_input.reshape.shape.extend([])
new_input.optional = True
# {
#   name: "top_p_min"
#   data_type: TYPE_FP32
#   dims: [ 1 ]
#   reshape: { shape: [ ] }
#   optional: true
# }
new_input = new_config.input.add()
new_input.name = 'top_p_min'
new_input.data_type = TYPE_FP32
new_input.dims.extend([1])
new_input.reshape.shape.extend([])
new_input.optional = True
# {
#   name: "top_p_reset_ids"
#   data_type: TYPE_UINT32
#   dims: [ 1 ]
#   reshape: { shape: [ ] }
#   optional: true
# }
new_input = new_config.input.add()
new_input.name = 'top_p_reset_ids'
new_input.data_type = TYPE_UINT32
new_input.dims.extend([1])
new_input.reshape.shape.extend([])
new_input.optional = True

# Outputs: these are all unchanged
new_config.output.extend(config.output)
# Instance group also unchanged
new_config.instance_group.extend(config.instance_group)

common_parameters = set([
    'tensor_para_size', 'pipeline_para_size', 'model_type',
    'model_checkpoint_path', 'enable_custom_all_reduce',
])
for parameter in config.parameters:
    if parameter not in common_parameters: continue
    new_config.parameters[parameter].string_value = config.parameters[parameter].string_value

# New parameters
new_config.parameters['data_type'].string_value = (
    'fp32' if config.parameters['is_half'].string_value == '0' else 'fp16'
)

# These parameters moved to config.ini in the weights directory
config_ini_params = {
    'model_name': 'model_name',
    'head_num': 'head_num',
    'size_per_head': 'size_per_head',
    'inter_size': 'inter_size',
    'decoder_layers': 'num_layer',
    'rotary_embedding': 'rotary_embedding',
    'vocab_size': 'vocab_size',
    'start_id': 'start_id',
    'end_id': 'end_id',
}
config_ini = ConfigParser()
config_ini.add_section('gptj')
for param in config_ini_params:
    config_ini['gptj'][config_ini_params[param]] = config.parameters[param].string_value
config_ini['gptj']['weight_data_type'] = 'fp32'

weights_dir = config.parameters['model_checkpoint_path'].string_value
# The weights dir in the config file may be remapped, e.g.
# /fastdata/mymodels/codegen-6B-mono-1gpu/fastertransformer/1/1-gpu
#   -> /model/fastertransformer/1/1-gpu
# Undo this remapping so we can find the config.ini file
# Find the 'fastertransformer' component of the path
orig_index = config_path.split(os.path.sep).index('fastertransformer')
# Find the 'fastertransformer' component of the weights dir
weights_index = weights_dir.split(os.path.sep).index('fastertransformer')
real_weights_dir = os.path.sep.join(
    config_path.split(os.path.sep)[:orig_index] +
    weights_dir.split(os.path.sep)[weights_index:]
)
config_ini_path = os.path.join(real_weights_dir, 'config.ini')

# Make backup copies of config.ini and config.pbtxt
os.rename(config_path, config_path + backup_ext)
if os.path.exists(config_ini_path):
    os.rename(config_ini_path, config_ini_path + backup_ext)

# Write out the new config files
with open(config_path, 'w') as f:
    f.write(text_format.MessageToString(new_config))
with open(config_ini_path, 'w') as f:
    config_ini.write(f)

# Write out the new version
with open(version_path, 'w') as f:
    print(new_version, file=f)

print(f'INFO: Successfully upgraded {model_dir} from {old_version} to {new_version}',
    file=sys.stderr)
