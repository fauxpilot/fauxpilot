FROM moyix/triton_with_ft:22.09

# Install dependencies: torch
RUN pip3 install -U torch --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip3 install -U transformers bitsandbytes accelerate