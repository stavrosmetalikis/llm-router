#!/bin/bash
cd "$(dirname "$0")"
git clone https://github.com/open-compress/claw-compactor.git

# Create and activate an isolated virtual environment
python3 -m venv .venv

# Install dependencies into the virtual environment using the venv's pip directly
.venv/bin/pip install -e claw-compactor[dev,accurate]
.venv/bin/pip install flask

echo "Done. Run the sidecar using the virtual environment interpreter:"
echo ".venv/bin/python main.py"
