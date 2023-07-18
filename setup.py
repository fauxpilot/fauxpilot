#!/usr/bin/env python3
import os
import subprocess
import tarfile
import sys
import importlib
import shutil


def check_missing_dependencies(dependencies):
    missing_dependencies = []
    for dependency in dependencies:
        try:
            importlib.import_module(dependency)
        except ImportError:
            missing_dependencies.append(dependency)

    if missing_dependencies:
        print("Missing Python dependencies: " + str(missing_dependencies))
        print("Please run:  pip install -r requirements.txt")
        exit(1)


dependencies = ['pyzstd', 'dotenv']
missing_dependencies = check_missing_dependencies(dependencies=dependencies)

import pyzstd
from dotenv import load_dotenv


def decompress_tar_zst(file_path, output_dir, model_dir):
    print("Decompressing model. This may take a few minutes.")
    # Create the output directory if it doesn't exist
    try:
        os.makedirs(output_dir, exist_ok=False)
    except OSError as e:
        print("Directory may already exist: " + output_dir)

    # Open the .tar.zst file for reading
    with open(file_path, 'rb') as file:
        # Decompress the file data
        decompressed_data = pyzstd.decompress(file.read())

    # Create a temporary file to hold the decompressed data
    temp_file_path = os.path.join(output_dir, '.temp.tar')

    # Write the decompressed data to the temporary file
    with open(temp_file_path, 'wb') as temp_file:
        temp_file.write(decompressed_data)

    # Open the temporary .tar file and extract its contents to the output directory
    with tarfile.open(temp_file_path, 'r') as tar:
        tar.extractall(output_dir)
        print("Model saved to: " + output_dir)
        shutil.copy(os.path.join(output_dir, 'fastertransformer/config.pbtxt'), os.path.join(model_dir, 'fastertransformer/config.pbtxt'))

    # Remove the temporary .tar file
    os.remove(temp_file_path)


def check_dependency(dependency):
    try:
        subprocess.check_output(["where" if os.name == "nt" else "which", dependency])
    except subprocess.CalledProcessError:
        print(f"Dependency '{dependency}' not found.")
        exit(1)


# Check if .env file exists
if os.path.isfile(".env"):
    delete_env = input(".env already exists, do you want to delete .env and recreate it? [y/n] ")
    if delete_env.lower() in ["y", "yes"]:
        print("Deleting .env")
        os.remove(".env")
    else:
        print("Exiting")
        exit(0)

# Check dependencies
check_dependency("curl")
check_dependency("docker")

############### Common configuration ###############

# Read number of GPUs
num_gpus = input("Enter number of GPUs [1]: ") or "1"

# External port for the API
api_external_port = input("External port for the API [5000]: ") or "5000"

# Address for Triton
triton_host = input("Address for Triton [triton]: ") or "triton"

# Port of Triton host
triton_port = input("Port of Triton host [8001]: ") or "8001"

# Read models root directory
joined_path = os.path.join(os.getcwd(), "models")
models_root_dir = input(f"Where do you want to save your models [" + joined_path + "]? ") or joined_path
models_root_dir = os.path.abspath(os.path.expanduser(models_root_dir))
os.makedirs(models_root_dir, exist_ok=True)

# Write .env
with open(".env", "a") as env_file:
    env_file.write(f"NUM_GPUS={num_gpus}\n")
    env_file.write(f"GPUS={','.join(str(i) for i in range(int(num_gpus)))}\n")
    env_file.write(f"API_EXTERNAL_PORT={api_external_port}\n")
    env_file.write(f"TRITON_HOST={triton_host}\n")
    env_file.write(f"TRITON_PORT={triton_port}\n")


############### Backend specific configuration ###############

def fastertransformer_backend():
    print("Models available:")
    print("[1] codegen-350M-mono (2GB total VRAM required; Python-only)")
    print("[2] codegen-350M-multi (2GB total VRAM required; multi-language)")
    print("[3] codegen-2B-mono (7GB total VRAM required; Python-only)")
    print("[4] codegen-2B-multi (7GB total VRAM required; multi-language)")
    print("[5] codegen-6B-mono (13GB total VRAM required; Python-only)")
    print("[6] codegen-6B-multi (13GB total VRAM required; multi-language)")
    print("[7] codegen-16B-mono (32GB total VRAM required; Python-only)")
    print("[8] codegen-16B-multi (32GB total VRAM required; multi-language)")

    # Read user choice
    model_num = input("Enter your choice [6]: ") or "6"

    # Convert model number to model name
    model_names = [
        "codegen-350M-mono",
        "codegen-350M-multi",
        "codegen-2B-mono",
        "codegen-2B-multi",
        "codegen-6B-mono",
        "codegen-6B-multi",
        "codegen-16B-mono",
        "codegen-16B-multi"
    ]
    model = model_names[int(model_num) - 1] if 1 <= int(model_num) <= len(model_names) else "codegen-6B-multi"

    with open(".env", "a") as env_file:
        env_file.write(f"MODEL={model}\n")
        env_file.write(f"MODEL_DIR={os.path.join(models_root_dir, f'{model}-{num_gpus}gpu')}\n")

    model_dir = os.path.join(models_root_dir, f"{model}-{num_gpus}gpu")
    if os.path.isdir(model_dir):
        print(model_dir)
        print(f"Converted model for {model}-{num_gpus}gpu already exists.")
        reuse_choice = input("Do you want to re-use it? y/n: ") or "y"
        if reuse_choice.lower() in ["y", "yes"]:
            download_model = "n"
            print("Re-using model")
        else:
            download_model = "y"
            shutil.rmtree(model_dir)
    else:
        download_model = "y"

    if download_model.lower() in ["y", "yes"]:
        if int(num_gpus) <= 2:
            print("Downloading the model from HuggingFace, this will take a while...")
            script_dir = os.path.dirname(os.path.realpath(__file__))
            dest = f"{model}-{num_gpus}gpu"
            archive = os.path.join(models_root_dir, f"{dest}.tar.zst")
            save_dir = os.path.normpath(os.path.join(script_dir, "converter/models", dest))
            print(models_root_dir)

            shutil.copytree(save_dir, models_root_dir, dirs_exist_ok=True)
            if not os.path.exists(archive):
                subprocess.run(
                    ["curl", "-L",
                     f"https://huggingface.co/moyix/{model}-gptj/resolve/main/{model}-{num_gpus}gpu.tar.zst",
                     "-o", archive])
            decompress_tar_zst(file_path=archive, output_dir=models_root_dir, model_dir=model_dir)
            if os.path.isfile(archive):
                redownload_archive = input("Remove downloaded archive file?: [y/n] ") or "y"
                if redownload_archive.lower() in ["yes", "y"]:
                    os.remove(archive)
        else:
            print("Downloading and converting the model, this will take a while...")
            subprocess.run(["docker", "run", "--rm", "-v", f"{models_root_dir}:/models", "-e", f"MODEL={model}", "-e",
                            f"NUM_GPUS={num_gpus}", "moyix/model_converter:latest"])

    hf_cache_dir = os.path.join(os.getcwd(), ".hf_cache")
    os.makedirs(hf_cache_dir, exist_ok=True)
    with open(".env", "a") as env_file:
        env_file.write(f"HF_CACHE_DIR={hf_cache_dir}\n")


def python_backend():
    print("Models available:")
    print("[1] codegen-350M-mono (1GB total VRAM required; Python-only)")
    print("[2] codegen-350M-multi (1GB total VRAM required; multi-language)")
    print("[3] codegen-2B-mono (4GB total VRAM required; Python-only)")
    print("[4] codegen-2B-multi (4GB total VRAM required; multi-language)")

    # Read user choice
    model_num = input("Enter your choice [4]: ") or "4"

    # Convert model number to model name
    model_names = [
        ("codegen-350M-mono", "Salesforce"),
        ("codegen-350M-multi", "Salesforce"),
        ("codegen-2B-mono", "Salesforce"),
        ("codegen-2B-multi", "Salesforce")
    ]
    model, org = model_names[int(model_num) - 1] if 1 <= int(model_num) <= len(model_names) else (
        "codegen-2B-multi", "Salesforce")

    share_hf_cache = input(
        "Do you want to share your huggingface cache between host and docker container? y/n [n]: ") or "n"
    if share_hf_cache.lower() in ["y", "yes"]:
        default_cache = os.path.normpath(os.path.expanduser("~/.cache/huggingface"))
        hf_cache_dir = input(
            "Enter your huggingface cache directory [" + default_cache + "]: ") or default_cache
    else:
        hf_cache_dir = os.path.join(os.getcwd(), ".hf_cache")

    use_int8 = input("Do you want to use int8? y/n [n]: ") or "n"
    use_int8 = "1" if use_int8.lower() in ["y", "yes"] else "0"

    with open(".env", "a") as env_file:
        env_file.write(f"MODEL=py-{model}\n")
        env_file.write(f"MODEL_DIR={os.path.join(models_root_dir, f'py-{org}-{model}')}\n")
        env_file.write(f"HF_CACHE_DIR={hf_cache_dir}\n")

    subprocess.run(
        [sys.executable, "./python_backend/init_model.py", "--model_name", model, "--org_name", org, "--model_dir",
         models_root_dir, "--use_int8", use_int8])
    load_dotenv('.env')
    command = "docker compose build || docker-compose build"
    subprocess.run(command, shell=True)


# Choose backend
print("Choose your backend:")
print("[1] FasterTransformer backend (faster, but limited models)")
print("[2] Python backend (slower, but more models, and allows loading with int8)")
backend_num = input("Enter your choice [1]: ") or "1"

if backend_num == "2":
    python_backend()
else:
    fastertransformer_backend()

run_fauxpilot = input("Config complete, do you want to run FauxPilot? [y/n] ") or "y"
if run_fauxpilot.lower() in ["y", "yes"]:
    subprocess.run([sys.executable, "launch.py"])
else:
    print("You can run python launch.py to start the FauxPilot server.")
    exit(0)
