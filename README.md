
# FauxPilot

This is an attempt to build a locally hosted alternative to [GitHub Copilot](https://copilot.github.com/). It uses the [SalesForce CodeGen](https://github.com/salesforce/CodeGen) models inside of NVIDIA's [Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server) with the [FasterTransformer backend](https://github.com/triton-inference-server/fastertransformer_backend/).

<p align="right">
  <img width="50%" align="right" src="./img/fauxpilot.png">
</p>

## Prerequisites

You'll need:

* Docker
* `docker compose` >= 1.28
* An NVIDIA GPU with Compute Capability >= 6.0 and enough VRAM to run the model you want.
* [`nvidia-docker`](https://github.com/NVIDIA/nvidia-docker)
* `curl` and `zstd` for downloading and unpacking the models.

Note that the VRAM requirements listed by `setup.sh` are *total* -- if you have multiple GPUs, you can split the model across them. So, if you have two NVIDIA RTX 3080 GPUs, you *should* be able to run the 6B model by putting half on each GPU.


## Support and Warranty

lmao

Okay, fine, we now have some minimal information on [the wiki](https://github.com/moyix/fauxpilot/wiki) and a [discussion forum](https://github.com/moyix/fauxpilot/discussions) where you can ask questions. Still no formal support or warranty though!



## Setup

This section describes how to install a Fauxpilot server and clients.

### Setting up a FauxPilot Server

Run the setup script to choose a model to use. This will download the model from [Huggingface/Moyix](https://huggingface.co/Moyix) in GPT-J format and then convert it for use with FasterTransformer.

Please refer to [How to set-up a FauxPilot server](documentation/server.md).


### Client configuration for FauxPilot

We offer some ways to connect to FauxPilot Server. For example, you can create a client by how to open the Openai API, Copilot Plugin, REST API.

Please refer to [How to set-up a client](documentation/client.md).


## Terminology
 * API: Application Programming Interface
 * CC: Compute Capability
 * CUDA: Compute Unified Device Architecture
 * FT: Faster Transformer
 * JSON: JavaScript Object Notation 
 * gRPC: Remote Procedure call by Google
 * GPT-J: A transformer model trained using Ben Wang's Mesh Transformer JAX 
 * REST: REpresentational State Transfer
