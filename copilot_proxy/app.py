#!/usr/bin/env python

import time
import random
import string
from flask import Flask, request
import numpy as np
from flask_wtf.csrf import CSRFProtect
np.finfo(np.dtype("float32"))
np.finfo(np.dtype("float64"))
import tritonclient.grpc as client_util
import json
from tritonclient.utils import np_to_triton_dtype
from tokenizers import Tokenizer

# These are the VSCode settings to tell Copilot to use a different server:
#     "github.copilot.advanced": {
#         "debug.overrideEngine": "codeai",
#         "debug.testOverrideProxyUrl": "http://9.135.120.183:5000",
#         "debug.overrideProxyUrl": "http://9.135.120.183:5000"
#     }

PAD_CHAR = 50256

# Max number of tokens the model can handle
MAX_MODEL_LEN = 2048

def to_word_list_format(word_dict, tokenizer):
    flat_ids = []
    offsets = []
    for word_dict_item in word_dict:
        item_flat_ids = []
        item_offsets = []

        for word in word_dict_item:
            ids = tokenizer.encode(word).ids

            if len(ids) == 0:
                continue

            item_flat_ids += ids
            item_offsets.append(len(ids))

            # Hack, can we do this better?
            if word == '\n\n':
                item_flat_ids += [198,198]
                item_offsets.append(2)

        flat_ids.append(np.array(item_flat_ids))
        offsets.append(np.cumsum(np.array(item_offsets)))

    pad_to = max(1, max(len(ids) for ids in flat_ids))

    for i, (ids, offs) in enumerate(zip(flat_ids, offsets)):
        flat_ids[i] = np.pad(ids, (0, pad_to - len(ids)), constant_values=0)
        offsets[i] = np.pad(offs, (0, pad_to - len(offs)), constant_values=-1)

    return np.array([flat_ids, offsets], dtype="int32").transpose((1, 0, 2))

def prepare_tensor(name, input):
    t = client_util.InferInput(
        name, input.shape, np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t

class CodeGenProxy:
    def __init__(self, max_time=10, host='localhost', port=8001, verbose=False):
        self.max_time = max_time
        self.url = f'{host}:{port}'
        self.verbose = verbose
        self.tokenizer = Tokenizer.from_file('/python-docker/cgtok/tokenizer.json')
        self.client = client_util.InferenceServerClient(self.url, verbose=self.verbose)

    def trim_with_stopwords(self, output, stopwords):
        for w in sorted(stopwords, key=len, reverse=True):
            if output.endswith(w):
                output = output[:-len(w)]
                break
        return output

    def generate(self, data):
        model_name = "fastertransformer"
        prompt = data['prompt']
        n = data.get('n',1)
        input_start_ids = np.expand_dims(self.tokenizer.encode(prompt).ids,0)
        input_start_ids = np.repeat(input_start_ids, n, axis=0).astype(np.uint32)
        prompt_len = input_start_ids.shape[1]
        input_len = prompt_len * np.ones([input_start_ids.shape[0], 1]).astype(np.uint32)
        max_tokens = data.get('max_tokens', 16)
        if max_tokens + input_len[0][0] > MAX_MODEL_LEN:
            raise ValueError("Max tokens + prompt length exceeds maximum model length")
        output_len = np.ones_like(input_len).astype(np.uint32) * max_tokens
        num_logprobs = data.get('logprobs', -1)
        want_logprobs = num_logprobs > 0

        temperature = data.get('temperature', 0.2)
        if temperature == 0.0:
            temperature = 1.0
            top_k = 1
        else:
            top_k = data.get('top_k', 0)
        
        top_p = data.get('top_p', 1.0)
        repetition_penalty = data.get('frequency_penalty', 1.0)
        runtime_top_k = top_k * np.ones([input_start_ids.shape[0], 1]).astype(np.uint32)
        runtime_top_p = top_p * np.ones([input_start_ids.shape[0], 1]).astype(np.float32)
        beam_search_diversity_rate = 0.0 * np.ones([input_start_ids.shape[0], 1]).astype(np.float32)
        random_seed = np.random.randint(0, 2**31-1, (input_start_ids.shape[0], 1), dtype=np.int32)
        temperature = temperature * np.ones([input_start_ids.shape[0], 1]).astype(np.float32)
        len_penalty = 1.0 * np.ones([input_start_ids.shape[0], 1]).astype(np.float32)
        repetition_penalty = repetition_penalty * np.ones([input_start_ids.shape[0], 1]).astype(np.float32)
        is_return_log_probs = want_logprobs * np.ones([input_start_ids.shape[0], 1]).astype(np.bool_)
        beam_width = (1 * np.ones([input_start_ids.shape[0], 1])).astype(np.uint32)
        start_ids = PAD_CHAR * np.ones([input_start_ids.shape[0], 1]).astype(np.uint32)
        end_ids = PAD_CHAR * np.ones([input_start_ids.shape[0], 1]).astype(np.uint32)

        stop_words = data.get('stop', [])
        if stop_words:
            stop_word_list = np.repeat(to_word_list_format([stop_words], self.tokenizer), input_start_ids.shape[0], axis=0)
        else:
            stop_word_list = np.concatenate([np.zeros([input_start_ids.shape[0], 1, 1]).astype(
                np.int32), (-1 * np.ones([input_start_ids.shape[0], 1, 1])).astype(np.int32)], axis=1)

        # Not used
        bad_words_list = np.concatenate([np.zeros([input_start_ids.shape[0], 1, 1]).astype(
                np.int32), (-1 * np.ones([input_start_ids.shape[0], 1, 1])).astype(np.int32)], axis=1)
        
        inputs = [
            prepare_tensor("input_ids", input_start_ids),
            prepare_tensor("input_lengths", input_len),
            prepare_tensor("request_output_len", output_len),
            prepare_tensor("runtime_top_k", runtime_top_k),
            prepare_tensor("runtime_top_p", runtime_top_p),
            prepare_tensor("beam_search_diversity_rate", beam_search_diversity_rate),
            prepare_tensor("random_seed", random_seed),
            prepare_tensor("temperature", temperature),
            prepare_tensor("len_penalty", len_penalty),
            prepare_tensor("repetition_penalty", repetition_penalty),
            prepare_tensor("is_return_log_probs", is_return_log_probs),
            prepare_tensor("beam_width", beam_width),
            prepare_tensor("start_id", start_ids),
            prepare_tensor("end_id", end_ids),
            prepare_tensor("bad_words_list", bad_words_list),
            prepare_tensor("stop_words_list", stop_word_list),
        ]

        result = self.client.infer(model_name, inputs)
        output_data = result.as_numpy("output_ids")
        if output_data is None:
            raise RuntimeError("No output data")

        # All of these squeeze(1)s are to remove the beam width dimension.
        output_data = output_data.squeeze(1)
        if want_logprobs:
            lp_data = result.as_numpy("output_log_probs").squeeze(1)
            # clp_data = result.as_numpy("cum_log_probs").squeeze(1)
        else:
            lp_data = [None]*output_data.shape[0]
        sequence_lengths = result.as_numpy("sequence_length").squeeze(1)
        gen_len = sequence_lengths - input_len.squeeze(1)
        
        decoded = self.tokenizer.decode_batch([out[prompt_len:prompt_len+g] for g,out in zip(gen_len,output_data)])
        trimmed = [self.trim_with_stopwords(d, stop_words) for d in decoded]

        choices = []
        for i, (text, tokens, lps, g) in enumerate(zip(trimmed, output_data, lp_data, gen_len)):
            reason = "length" if max_tokens == g else "stop"            
            if lps is not None:
                tokens_str = [self.tokenizer.decode([t]) for t in tokens[prompt_len:prompt_len+g]]
                offsets = [len(prompt)] + (np.cumsum([len(t) for t in tokens_str]) + len(prompt)).tolist()[:-1]

                # Fake some log probs for top_logprobs
                top_logprobs = []                
                for i,t in enumerate(tokens_str):
                    fakedict = {}
                    top_token_lp = float(lps[i])
                    fakedict[t] = top_token_lp
                    while len(fakedict) < num_logprobs:
                        random_token = random.randint(0, self.tokenizer.get_vocab_size()-1)
                        random_token_str = self.tokenizer.decode([random_token])
                        if random_token_str in fakedict: continue
                        random_token_lp = top_token_lp - random.random()
                        fakedict[random_token_str] = random_token_lp
                    top_logprobs.append(fakedict)

                lpdict = {
                    'token_logprobs': lps.tolist(),
                    'top_logprobs': top_logprobs, # TODO
                    'tokens': tokens_str,
                    'text_offset': offsets,
                }
            else:
                lpdict = None
            
            choice = {
                'text': text,
                'index': i,
                'finish_reason': reason,
                'logprobs': lpdict,
            }
            choices.append(choice)
        
        completion = {
            'id': None, # fill in
            'model': 'codegen',
            'object': 'text_completion',
            'created': int(time.time()),
            'choices': None, # fill in
            'usage': {
                'completion_tokens': int(gen_len.sum()),
                'prompt_tokens': int(prompt_len),
                'total_tokens': int(gen_len.sum() + prompt_len),
            }
        }
        
        return completion, choices

    def random_completion_id(self):
        return 'cmpl-' + ''.join(random.choice(string.ascii_letters+string.digits) for _ in range(29))

    def streamed_response(self, completion, choices):
        for c in choices:
            completion['id'] = self.random_completion_id()
            completion['choices'] = [c]
            yield f'data: {json.dumps(completion)}\n\n'
        yield 'data: [DONE]\n\n'

    def non_streamed_response(self, completion, choices):
        completion['id'] = self.random_completion_id()
        completion['choices'] = choices
        return json.dumps(completion)

    def __call__(self, data):
        st = time.time()
        completion, choices = self.generate(data)
        ed = time.time()
        print(f"Returned completion in {(ed-st)*1000} ms")
        if data.get('stream', False):
            return self.streamed_response(completion, choices)
        else:
            return self.non_streamed_response(completion, choices)

codegen = CodeGenProxy(host='triton')
# OpenRefactory Warning: The 'Flask' method creates a Flask app
# without Cross-Site Request Forgery (CSRF) protection.
app = Flask(__name__)
CSRFProtect(app)

@app.route("/v1/engines/codegen/completions", methods=["POST"])
def completions():
    if request.method == "POST":
        data = json.loads(request.data)
        print(data)
        if data.get('stream', False):
            return app.response_class(
                response=codegen(data),
                status=200,
                mimetype='text/event-stream'
            )
        else:
            return app.response_class(
                response=codegen(data),
                status=200,
                mimetype='application/json'
            )

if __name__ == "__main__":
    app.run(debug=True)
