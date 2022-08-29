# FauxPilot

This is an attempt to build a locally hosted version of [GitHub Copilot](https://copilot.github.com/). It uses the [SalesForce CodeGen](https://github.com/salesforce/CodeGen) models inside of NVIDIA's [Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server) with the [FasterTransformer backend](https://github.com/triton-inference-server/fastertransformer_backend/).

## Prerequisites

You'll need:

* Docker
* `docker compose` >= 1.28
* An NVIDIA GPU with Compute Capability >= 7.0 and enough VRAM to run the model you want.
* [`nvidia-docker`](https://github.com/NVIDIA/nvidia-docker)
* `curl` and `zstd` for downloading and unpacking the models.

Note that the VRAM requirements listed by `setup.sh` are *total* -- if you have multiple GPUs, you can split the model across them. So, if you have two NVIDIA RTX 3080 GPUs, you *should* be able to run the 6B model by putting half on each GPU.

## Support and Warranty

lmao

## Setup

Run the setup script to choose a model to use. This will download the model from Huggingface and then convert it for use with FasterTransformer.

```
$ ./setup.sh 
Models available:
[1] codegen-350M-mono (2GB total VRAM required; Python-only)
[2] codegen-350M-multi (2GB total VRAM required; multi-language)
[3] codegen-2B-mono (7GB total VRAM required; Python-only)
[4] codegen-2B-multi (7GB total VRAM required; multi-language)
[5] codegen-6B-mono (13GB total VRAM required; Python-only)
[6] codegen-6B-multi (13GB total VRAM required; multi-language)
[7] codegen-16B-mono (32GB total VRAM required; Python-only)
[8] codegen-16B-multi (32GB total VRAM required; multi-language)
Enter your choice [6]: 2
Enter number of GPUs [1]: 1
Where do you want to save the model [/home/moyix/git/fauxpilot/models]? /fastdata/mymodels
Downloading and converting the model, this will take a while...
Converting model codegen-350M-multi with 1 GPUs
Loading CodeGen model
Downloading config.json: 100%|██████████| 996/996 [00:00<00:00, 1.25MB/s]
Downloading pytorch_model.bin: 100%|██████████| 760M/760M [00:11<00:00, 68.3MB/s] 
Creating empty GPTJ model
Converting...
Conversion complete.
Saving model to codegen-350M-multi-hf...

=============== Argument ===============
saved_dir: /models/codegen-350M-multi-1gpu/fastertransformer/1
in_file: codegen-350M-multi-hf
trained_gpu_num: 1
infer_gpu_num: 1
processes: 4
weight_data_type: fp32
========================================
transformer.wte.weight
transformer.h.0.ln_1.weight
[... more conversion output trimmed ...]
transformer.ln_f.weight
transformer.ln_f.bias
lm_head.weight
lm_head.bias
Done! Now run ./launch.sh to start the FauxPilot server.
```

Then you can just run `./launch.sh`:

```
$ ./launch.sh 
[+] Running 2/0
 ⠿ Container fauxpilot-triton-1         Created                                                                                                                                                                                                                                                                                             0.0s
 ⠿ Container fauxpilot-copilot_proxy-1  Created                                                                                                                                                                                                                                                                                             0.0s
Attaching to fauxpilot-copilot_proxy-1, fauxpilot-triton-1
fauxpilot-triton-1         | 
fauxpilot-triton-1         | =============================
fauxpilot-triton-1         | == Triton Inference Server ==
fauxpilot-triton-1         | =============================
fauxpilot-triton-1         | 
fauxpilot-triton-1         | NVIDIA Release 22.06 (build 39726160)
fauxpilot-triton-1         | Triton Server Version 2.23.0
fauxpilot-triton-1         | 
fauxpilot-triton-1         | Copyright (c) 2018-2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
fauxpilot-triton-1         | 
fauxpilot-triton-1         | Various files include modifications (c) NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
fauxpilot-triton-1         | 
fauxpilot-triton-1         | This container image and its contents are governed by the NVIDIA Deep Learning Container License.
fauxpilot-triton-1         | By pulling and using the container, you accept the terms and conditions of this license:
fauxpilot-triton-1         | https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license
fauxpilot-copilot_proxy-1  | WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
fauxpilot-copilot_proxy-1  |  * Debug mode: off
fauxpilot-copilot_proxy-1  |  * Running on all addresses (0.0.0.0)
fauxpilot-copilot_proxy-1  |    WARNING: This is a development server. Do not use it in a production deployment.
fauxpilot-copilot_proxy-1  |  * Running on http://127.0.0.1:5000
fauxpilot-copilot_proxy-1  |  * Running on http://172.18.0.3:5000 (Press CTRL+C to quit)
fauxpilot-triton-1         | 
fauxpilot-triton-1         | ERROR: This container was built for NVIDIA Driver Release 515.48 or later, but
fauxpilot-triton-1         |        version  was detected and compatibility mode is UNAVAILABLE.
fauxpilot-triton-1         | 
fauxpilot-triton-1         |        [[]]
fauxpilot-triton-1         | 
fauxpilot-triton-1         | I0803 01:51:02.690042 93 pinned_memory_manager.cc:240] Pinned memory pool is created at '0x7f6104000000' with size 268435456
fauxpilot-triton-1         | I0803 01:51:02.690461 93 cuda_memory_manager.cc:105] CUDA memory pool is created on device 0 with size 67108864
fauxpilot-triton-1         | I0803 01:51:02.692434 93 model_repository_manager.cc:1191] loading: fastertransformer:1
fauxpilot-triton-1         | I0803 01:51:02.936798 93 libfastertransformer.cc:1226] TRITONBACKEND_Initialize: fastertransformer
fauxpilot-triton-1         | I0803 01:51:02.936818 93 libfastertransformer.cc:1236] Triton TRITONBACKEND API version: 1.10
fauxpilot-triton-1         | I0803 01:51:02.936821 93 libfastertransformer.cc:1242] 'fastertransformer' TRITONBACKEND API version: 1.10
fauxpilot-triton-1         | I0803 01:51:02.936850 93 libfastertransformer.cc:1274] TRITONBACKEND_ModelInitialize: fastertransformer (version 1)
fauxpilot-triton-1         | W0803 01:51:02.937855 93 libfastertransformer.cc:149] model configuration:
fauxpilot-triton-1         | {
[... lots more output trimmed ...]
fauxpilot-triton-1         | I0803 01:51:04.711929 93 libfastertransformer.cc:321] After Loading Model:
fauxpilot-triton-1         | I0803 01:51:04.712427 93 libfastertransformer.cc:537] Model instance is created on GPU NVIDIA RTX A6000
fauxpilot-triton-1         | I0803 01:51:04.712694 93 model_repository_manager.cc:1345] successfully loaded 'fastertransformer' version 1
fauxpilot-triton-1         | I0803 01:51:04.712841 93 server.cc:556] 
fauxpilot-triton-1         | +------------------+------+
fauxpilot-triton-1         | | Repository Agent | Path |
fauxpilot-triton-1         | +------------------+------+
fauxpilot-triton-1         | +------------------+------+
fauxpilot-triton-1         | 
fauxpilot-triton-1         | I0803 01:51:04.712916 93 server.cc:583] 
fauxpilot-triton-1         | +-------------------+-----------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------+
fauxpilot-triton-1         | | Backend           | Path                                                                        | Config                                                                                                                                                         |
fauxpilot-triton-1         | +-------------------+-----------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------+
fauxpilot-triton-1         | | fastertransformer | /opt/tritonserver/backends/fastertransformer/libtriton_fastertransformer.so | {"cmdline":{"auto-complete-config":"false","min-compute-capability":"6.000000","backend-directory":"/opt/tritonserver/backends","default-max-batch-size":"4"}} |
fauxpilot-triton-1         | +-------------------+-----------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------+
fauxpilot-triton-1         | 
fauxpilot-triton-1         | I0803 01:51:04.712959 93 server.cc:626] 
fauxpilot-triton-1         | +-------------------+---------+--------+
fauxpilot-triton-1         | | Model             | Version | Status |
fauxpilot-triton-1         | +-------------------+---------+--------+
fauxpilot-triton-1         | | fastertransformer | 1       | READY  |
fauxpilot-triton-1         | +-------------------+---------+--------+
fauxpilot-triton-1         | 
fauxpilot-triton-1         | I0803 01:51:04.738989 93 metrics.cc:650] Collecting metrics for GPU 0: NVIDIA RTX A6000
fauxpilot-triton-1         | I0803 01:51:04.739373 93 tritonserver.cc:2159] 
fauxpilot-triton-1         | +----------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
fauxpilot-triton-1         | | Option                           | Value                                                                                                                                                                                        |
fauxpilot-triton-1         | +----------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
fauxpilot-triton-1         | | server_id                        | triton                                                                                                                                                                                       |
fauxpilot-triton-1         | | server_version                   | 2.23.0                                                                                                                                                                                       |
fauxpilot-triton-1         | | server_extensions                | classification sequence model_repository model_repository(unload_dependents) schedule_policy model_configuration system_shared_memory cuda_shared_memory binary_tensor_data statistics trace |
fauxpilot-triton-1         | | model_repository_path[0]         | /model                                                                                                                                                                                       |
fauxpilot-triton-1         | | model_control_mode               | MODE_NONE                                                                                                                                                                                    |
fauxpilot-triton-1         | | strict_model_config              | 1                                                                                                                                                                                            |
fauxpilot-triton-1         | | rate_limit                       | OFF                                                                                                                                                                                          |
fauxpilot-triton-1         | | pinned_memory_pool_byte_size     | 268435456                                                                                                                                                                                    |
fauxpilot-triton-1         | | cuda_memory_pool_byte_size{0}    | 67108864                                                                                                                                                                                     |
fauxpilot-triton-1         | | response_cache_byte_size         | 0                                                                                                                                                                                            |
fauxpilot-triton-1         | | min_supported_compute_capability | 6.0                                                                                                                                                                                          |
fauxpilot-triton-1         | | strict_readiness                 | 1                                                                                                                                                                                            |
fauxpilot-triton-1         | | exit_timeout                     | 30                                                                                                                                                                                           |
fauxpilot-triton-1         | +----------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
fauxpilot-triton-1         | 
fauxpilot-triton-1         | I0803 01:51:04.740423 93 grpc_server.cc:4587] Started GRPCInferenceService at 0.0.0.0:8001
fauxpilot-triton-1         | I0803 01:51:04.740608 93 http_server.cc:3303] Started HTTPService at 0.0.0.0:8000
fauxpilot-triton-1         | I0803 01:51:04.781561 93 http_server.cc:178] Started Metrics Service at 0.0.0.0:8002
```

## API

Once everything is up and running, you should have a server listening for requests on `http://localhost:5000`. You can now talk to it using the standard [OpenAI API](https://beta.openai.com/docs/api-reference/) (although the full API isn't implemented yet). For example, from Python, using the [OpenAI Python bindings](https://github.com/openai/openai-python):

```python
$ ipython
Python 3.8.10 (default, Mar 15 2022, 12:22:08) 
Type 'copyright', 'credits' or 'license' for more information
IPython 8.2.0 -- An enhanced Interactive Python. Type '?' for help.

In [1]: import openai

In [2]: openai.api_key = 'dummy'

In [3]: openai.api_base = 'http://127.0.0.1:5000/v1'

In [4]: result = openai.Completion.create(engine='codegen', prompt='def hello', max_tokens=16, temperature=0.1, stop=["\n\n"])

In [5]: result
Out[5]: 
<OpenAIObject text_completion id=cmpl-6hqu8Rcaq25078IHNJNVooU4xLY6w at 0x7f602c3d2f40> JSON: {
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "logprobs": null,
      "text": "() {\n    return \"Hello, World!\";\n}"
    }
  ],
  "created": 1659492191,
  "id": "cmpl-6hqu8Rcaq25078IHNJNVooU4xLY6w",
  "model": "codegen",
  "object": "text_completion",
  "usage": {
    "completion_tokens": 15,
    "prompt_tokens": 2,
    "total_tokens": 17
  }
}
```

## Copilot Plugin

Perhaps more excitingly, you can configure the official [VSCode Copilot plugin](https://marketplace.visualstudio.com/items?itemName=GitHub.copilot) to use your local server. Just edit your `settings.json` to add:

```json
    "github.copilot.advanced": {
        "debug.overrideEngine": "codegen",
        "debug.testOverrideProxyUrl": "http://localhost:5000",
        "debug.overrideProxyUrl": "http://localhost:5000"
    }
```

And you should be able to use Copilot with your own locally hosted suggestions! Of course, probably a lot of stuff is subtly broken. In particular, the probabilities returned by the server are partly fake. Fixing this would require changing FasterTransformer so that it can return log-probabilities for the top k tokens rather that just the chosen token.

Have fun!
