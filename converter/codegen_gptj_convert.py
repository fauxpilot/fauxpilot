#!/usr/bin/env python

import argparse
import torch
from transformers import GPTJForCausalLM, GPTJConfig
# Note: these need the git version of Transformers as of 7/22/2022
from transformers import CodeGenTokenizer, CodeGenForCausalLM  # noqa: F401
from transformers import CODEGEN_PRETRAINED_MODEL_ARCHIVE_LIST

CODEGEN_2_LIST = [
    "Salesforce/codegen2-1B",
    "Salesforce/codegen2-3_7B",
    "Salesforce/codegen2-7B",
    "Salesforce/codegen2-16B"
]
convertable_models = CODEGEN_PRETRAINED_MODEL_ARCHIVE_LIST + CODEGEN_2_LIST

parser = argparse.ArgumentParser('Convert SalesForce CodeGen model to GPT-J')
parser.add_argument('--code_model',
                    choices=convertable_models, default='Salesforce/codegen-350M-multi',
                    help='which SalesForce model to convert'
                    )
parser.add_argument('output_dir', help='where to store the converted model')
args = parser.parse_args()

print('Loading CodeGen model')
cg_model = CodeGenForCausalLM.from_pretrained(
    args.code_model, torch_dtype="auto", trust_remote_code=bool(args.code_model in CODEGEN_2_LIST)
)
cg_config = cg_model.config

# Create empty GPTJ model
print('Creating empty GPTJ model')
config = GPTJConfig(
    vocab_size=cg_config.vocab_size,
    n_positions=cg_config.n_positions,
    n_embd=cg_config.n_embd,
    n_layer=cg_config.n_layer,
    n_head=cg_config.n_head,
    rotary_dim=cg_config.rotary_dim,
    n_inner=cg_config.n_inner,
    activation_function=cg_config.activation_function,
    resid_pdrop=cg_config.resid_pdrop,
    embd_pdrop=cg_config.embd_pdrop,
    attn_pdrop=cg_config.attn_pdrop,
    layer_norm_epsilon=cg_config.layer_norm_epsilon,
    initializer_range=cg_config.initializer_range,
    scale_attn_weights=cg_config.scale_attn_weights,
    use_cache=cg_config.use_cache,
    bos_token_id=cg_config.bos_token_id,
    eos_token_id=cg_config.eos_token_id,
    torch_dtype=cg_config.torch_dtype,
)
# Fix tokenizer type
config.tokenizer_class = 'CodeGenTokenizer'

gptj_model = GPTJForCausalLM(config).half()
embed_dim = config.n_embd


def replace(model, weights, model_name):
    model.state_dict()[model_name].copy_(weights.detach())


def replace_by_name(dest_model, src_model, old_name, new_name):
    assert old_name in src_model.state_dict()
    assert new_name in dest_model.state_dict()
    replace(model=dest_model, weights=src_model.state_dict()[old_name], model_name=new_name)


print('Converting...')
# Copy weights from CodeGen model
with torch.no_grad():
    cg_model.eval()
    gptj_model.eval()

    for name, param in cg_model.named_parameters():
        # print(f'Converting {name}')
        # Handle the qkv weights separately because we need to split them
        if 'qkv_proj' in name:
            qkv_proj = param.detach().clone()
            mp_num = 4  # number of cores on their TPU I guess?
            # GPT-J and CodeGen slice up the qkv projection slightly differently.
            # After a great deal of pain, I figured out that this permutation on
            # the weights of the qkv_proj fixes it.
            base_permutation = [0, 3, 6, 9,
                                1, 4, 7, 10,
                                2, 5, 8, 11]
            if args.code_model in ["Salesforce/codegen2-1B", "Salesforce/codegen2-3_7B"]:
                # codegen2-1B and codegen2-3_7B were trained on different mp setting
                # see: https://github.com/fauxpilot/fauxpilot/issues/202
                base_permutation = [0, 3, 6, 9, 12, 15, 18, 21,
                                    1, 4, 7, 10, 13, 16, 19, 22,
                                    2, 5, 8, 11, 14, 17, 20, 23]
                mp_num = 8
            local_dim = embed_dim // mp_num
            permutation = torch.cat([torch.arange(i * local_dim, (i + 1) * local_dim) for i in base_permutation])
            # NB: we permute the *rows* here because the computation is xA.T
            new_qkv_proj = qkv_proj[permutation, :]
            # NB: the name QKV is misleading here; they are actually stored in
            #     the order QVK
            query, value, key = torch.split(new_qkv_proj, embed_dim, dim=0)
            replace(gptj_model, query, name.replace('qkv_proj', 'q_proj'))
            replace(gptj_model, key, name.replace('qkv_proj', 'k_proj'))
            replace(gptj_model, value, name.replace('qkv_proj', 'v_proj'))
        else:
            replace_by_name(dest_model=gptj_model, src_model=cg_model, old_name=name, new_name=name)

    print('Conversion complete.')
    print(f"Saving model to {args.output_dir}...")
    gptj_model.save_pretrained(args.output_dir)
