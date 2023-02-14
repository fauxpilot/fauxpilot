"""
Tests setup script (currently for Python backend)
"""

import os
import shutil
import signal
import subprocess
from pathlib import Path
from typing import Dict, Union

import pexpect
import pytest
import requests

curdir = Path(__file__).parent
root = curdir.parent.parent

test_models_dir = curdir.joinpath("models")


def setup_module():
    """
    Setup steps for tests in this module
    """
    assert root.joinpath("setup.sh").exists(), "setup.sh not found"
    if root.joinpath(".env").exists():
        shutil.move(str(root.joinpath(".env")), str(root.joinpath(".env.bak")))


def teardown_module():
    """
    Teardown steps for tests in this module
    """
    if root.joinpath(".env.bak").exists():
        shutil.move(str(root.joinpath(".env.bak")), str(root.joinpath(".env")))
    try:
        if test_models_dir:
            shutil.rmtree(test_models_dir)
    except Exception as exc:
        print(
            f"WARNING: Couldn't delete `{test_models_dir}` most likely due to permission issues."
            f"Run the tests with sudo to ensure this gets deleted automatically, or else delete manually. "
            f"Exception: {exc}"
        )


def enter_input(proc: pexpect.spawn, expect: str, input_s: str, timeout: int = 5) -> str:
    """
    Helper function to enter input for a given prompt. Returns consumed output.
    """

    try:
        proc.expect(expect, timeout=timeout)
    except pexpect.exceptions.TIMEOUT as exc:
        raise AssertionError(
            f"Timeout waiting for prompt: `{expect}`.\n"
            f"Output-before: `{proc.before}`\nOutput-after: `{proc.after}`"
        ) from exc

    after = str(proc.after)
    print(after)
    proc.sendline(input_s)
    return after


def run_common_setup_steps(n_gpus: int = 0) -> pexpect.spawn:
    """
    Helper function to run common setup steps.
    """
    proc = pexpect.pty_spawn.spawn(
        "./setup.sh 2>&1", encoding="utf-8", cwd=str(root),
    )
    proc.ignorecase = True

    enter_input(proc, r".*Enter number of GPUs[^:]+: ?", str(n_gpus))
    enter_input(proc, r".*port for the API[^:]+: ?", "5000")
    enter_input(proc, r".*Address for Triton[^:]+: ?", "triton")
    enter_input(proc, r".*Port of Triton[^:]+: ?", "8001")
    enter_input(proc, r".*save your models[^\?]+\? ?", str(test_models_dir.absolute()))

    return proc


def load_test_env():
    """
    Load test env vars
    """
    # Without loading default env vars, PATH won't be set correctly
    env = os.environ.copy()
    with open(curdir.joinpath("test.env"), "r", encoding="utf8") as test_env:
        for line in test_env:
            key, val = line.strip().split("=")
            env[key] = val
    return env


def run_inference(
    prompt: str, model: str = "py-model", port: int = 5000, return_all: bool = False,
    **kwargs
) -> Union[str, Dict]:
    """
    Invokes the copilot proxy with the given prompt and returns the completion
    """
    endpoint = f"http://localhost:{port}/v1/engines/codegen/completions"
    data = {
        "model": model,
        "prompt": prompt,
        "suffix": kwargs.get("suffix", ""),
        "max_tokens": kwargs.get("max_tokens", 16),
        "temperature": kwargs.get("temperature", 0.0),
        "top_p": kwargs.get("top_p", 1.0),
        "n": kwargs.get("n", 1),
        "stream": kwargs.get("stream", None),  # it's not true/false. It's None or not None :[
        "logprobs": kwargs.get("logprobs", 0),
        "stop": kwargs.get("stop", ""),
        "echo": kwargs.get("echo", True),
        "presence_penalty": kwargs.get("presence_penalty", 0.0),
        "frequency_penalty": kwargs.get("frequency_penalty", 0.0),
        "best_of": kwargs.get("best_of", 1),
        "logit_bias": kwargs.get("logit_bias", {}),
        "user": kwargs.get("user", "test"),
    }

    response = requests.post(endpoint, json=data)
    response.raise_for_status()

    if return_all:
        return response.json()
    return response.json()["choices"][0]["text"]


@pytest.mark.parametrize("n_gpus", [0])  # we don't have a GPU on CI
def test_python_backend(n_gpus: int):
    """
    Step 1: run $root/setup.sh while passing appropriate options via stdin
    Step 2: run docker-compose up with test.env sourced
    Step 3: call :5000 with appropriate request
    """
    proc = run_common_setup_steps(n_gpus)

    choices = enter_input(proc, r".*Choose your backend.*Enter your choice[^:]+: ?", "2")
    assert "[2] Python backend" in choices, "Option 2 should be Python backend"

    choices = enter_input(proc, r".*Models available:.*Enter your choice[^:]+: ?", "1")
    assert "[1] codegen-350M-mono" in choices, "Option 1 should be codegen-350M-mono"

    enter_input(proc, r".*share (your )?huggingface cache[^:]+: ?", "y")
    enter_input(proc, r".*cache directory[^:]+: ?", "")  # default
    enter_input(proc, r".*use int8[^:]+: ?", "n")
    enter_input(proc, r".*run FauxPilot\? \[y/n\] ", "n", timeout=120)

    # copy $root/.env to $curdir/test.env
    shutil.copy(str(root.joinpath(".env")), str(curdir.joinpath("test.env")))

    # run docker-compose up -f docker-compose-{without|with}-gpus.yml
    compose_file = f"docker-compose-with{'' if n_gpus > 0 else 'out'}-gpus.yaml"
    docker_proc = None
    try:
        docker_proc = pexpect.pty_spawn.spawn(
            f"docker compose -f {compose_file} up",
            encoding="utf-8",
            cwd=curdir,
            env=load_test_env(),
        )

        print("Waiting for API to be ready...")
        docker_proc.expect(r".*Started GRPCInferenceService at 0.0.0.0:8001", timeout=120)

        print("API ready, sending request...")

        # Simple test 1: hello world prompt without bells and whistles
        response = run_inference("def hello_world():\n", max_tokens=16, return_all=True)
        assert response["choices"][0]["text"].rstrip() == '    print("Hello World")\n\nhello_world()\n\n#'
        assert response["choices"][0]["finish_reason"] == "length"

    finally:
        if docker_proc is not None and docker_proc.isalive():
            docker_proc.kill(signal.SIGINT)

        # killing docker-compose process doesn't bring down the containers.
        # explicitly stop the containers:
        subprocess.run(["docker-compose", "-f", compose_file, "down"], cwd=curdir, check=True, env=load_test_env())
