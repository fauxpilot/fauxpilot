import json

import torch
import triton_python_backend_utils as pb_utils
# Using dlpack causes segfaults on some machines, so not using it for now
# But it supports zero copy transfer from triton tensors to torch tensors,
# so worth investigating further
# from torch.utils.dlpack import to_dlpack, from_dlpack
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


def pb2torch(request, name):
    tensor = pb_utils.get_input_tensor_by_name(request, name)
    return torch.from_numpy(tensor.as_numpy())
    # return from_dlpack(tensor.to_dlpack())


def torch2pb(name, tensor):
    return pb_utils.Tensor(name, tensor.numpy())
    # return pb_utils.Tensor.from_dlpack(name, to_dlpack(tensor))


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = model_config = json.loads(args["model_config"])
        org_name = model_config["parameters"].get("org_name", {"string_value": "Salesforce"})["string_value"]
        model_name = org_name + "/" + model_config["parameters"]["model_name"]["string_value"]

        def get_bool(x):
            return model_config["parameters"][x]["string_value"].lower() in ["1", "true"]

        is_half = get_bool("use_half") and torch.cuda.is_available()
        # This will make inference marginally slower, but will allow bigger models to fit in GPU
        int8 = get_bool("use_int8") and torch.cuda.is_available()
        auto_device_map = get_bool("use_auto_device_map") and torch.cuda.is_available()

        print("Cuda available?", torch.cuda.is_available())
        print(f"is_half: {is_half}, int8: {int8}, auto_device_map: {auto_device_map}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if is_half else ("auto" if torch.cuda.is_available() else torch.float32),
            load_in_8bit=int8,
            device_map="auto" if auto_device_map else None,
            low_cpu_mem_usage=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"Model {model_name} Loaded. Footprint: {self.model.get_memory_footprint()}")

        # set max_batch_size
        self.max_batch_size = 0  # model_config["max_batch_size"]

    def execute(self, requests):
        # TODO: don't just loop over requests. batch them up

        responses = []

        for request in requests:
            input_ids_torch = pb2torch(request, "input_ids")
            input_lengths_torch = pb2torch(request, "input_lengths")
            request_output_len_torch = pb2torch(request, "request_output_len")

            # Attention mask
            attention_mask = None
            if input_lengths_torch.min() != input_lengths_torch.max():
                attention_mask = torch.zeros(input_ids_torch.shape, dtype=torch.long)
                for i, l in enumerate(input_lengths_torch):
                    attention_mask[i, :l] = 1

            # Output length
            max_new_tokens = request_output_len_torch[0][0]

            top_k = pb_utils.get_input_tensor_by_name(request, "runtime_top_k").as_numpy().tolist()[0]
            top_p = pb_utils.get_input_tensor_by_name(request, "runtime_top_p").as_numpy().tolist()[0]
            temperature = pb_utils.get_input_tensor_by_name(request, "temperature").as_numpy().tolist()[0]
            # n_samples = pb_utils.get_input_tensor_by_name(request, "n")
            n_samples = 1  # TODO: client doesn't send this yet. instead it duplicates the request n times

            # Generate
            output_ids = self.model.generate(
                input_ids=input_ids_torch, attention_mask=attention_mask,
                max_new_tokens=max_new_tokens, do_sample=True, top_k=top_k, top_p=top_p, num_return_sequences=n_samples,
                temperature=temperature,
            )

            # client wants batch x beam_width x seq_len and we don't support beam_width yet
            output_ids = output_ids.unsqueeze(1)

            # create output tensors
            out_tensor_pb = torch2pb("output_ids", output_ids)

            # calculate sequence_length
            sequence_length = torch.zeros(output_ids.shape[:2], dtype=torch.int32)
            for i in range(output_ids.shape[0]):
                sequence_length[i, 0] = torch.sum(output_ids[i, 0] != self.model.config.eos_token_id).item()
            sequence_length_pb = torch2pb("sequence_length", sequence_length)

            # create response
            response = pb_utils.InferenceResponse([out_tensor_pb, sequence_length_pb])
            responses.append(response)

        return responses
