This section describes the Python scripts necessary for converting deep learning model files:

* `Dockerfile`: A Docker file used to construct an image based on Ubuntu 20.04 that includes the Transformer library.
  * `download_and_convert_model.sh`: A shell script that converts model codegen-6B-multi with the provided number of GPUs.
    * `codegen_gptj_convert.py`: A Python script for converting SalesForce CodeGen models to GPT-J (e.g., Salesforce/codegen-350M-multi).
    * `huggingface_gptj_convert.py`: A Python script for converting the HF model to the GPT-J format (e.g., GPTJForCausalLM model)
* `triton_config_gen.py`: A Python script that creates a config and weight file for running a Codgen model with Triton.
  * `config_template.pbtxt`: A template file for defining the config file's data format.

