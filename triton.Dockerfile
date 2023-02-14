FROM moyix/triton_with_ft:22.09

# Install dependencies: torch
RUN python3 -m pip install --disable-pip-version-check -U torch --extra-index-url https://download.pytorch.org/whl/cu116
RUN python3 -m pip install --disable-pip-version-check -U transformers bitsandbytes accelerate
