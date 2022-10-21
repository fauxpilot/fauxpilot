"""
A simple script that sets up the model directory of a given model for Triton.
"""

import argparse
import os
import shutil
from pathlib import Path
from string import Template

SCRIPT_DIR = Path(__file__).parent
CONFIG_TEMPLATE_PATH = os.path.join(SCRIPT_DIR, 'config_template.pbtxt')

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, required=True)
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--org_name", type=str, required=True)
parser.add_argument("--use_half", type=str, default="1")
parser.add_argument("--use_int8", type=str, default="0")
parser.add_argument("--use_auto_device_map", type=str, default="1")
args = parser.parse_args()


# Step1: Make model directory
model_dir_path = Path(os.path.join(Path(args.model_dir), f"py-{args.org_name}-{args.model_name}/py-model/1"))
model_dir_path.mkdir(parents=True, exist_ok=True)

# Step 2: copy model.py
shutil.copy(os.path.join(SCRIPT_DIR, 'model.py'), os.path.join(model_dir_path, 'model.py'))

# Step 3: Generate config.pbtxt
with open(CONFIG_TEMPLATE_PATH, 'r') as f:
    template = Template(f.read())

config = template.substitute(
    org_name=args.org_name,
    model_name=args.model_name,
    use_half=args.use_half,
    use_int8=args.use_int8,
    use_auto_device_map=args.use_auto_device_map,
)
with open(os.path.join(model_dir_path, '../config.pbtxt'), 'w') as f:
    f.write(config)
    print(f"Config written to {os.path.abspath(f.name)}")
