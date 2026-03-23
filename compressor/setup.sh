#!/bin/bash
cd "$(dirname "$0")"
git clone https://github.com/open-compress/claw-compactor.git

# Create and activate an isolated virtual environment
python -m venv .venv
source .venv/bin/activate || source .venv/Scripts/activate

# Install dependencies into the virtual environment
pip install -e claw-compactor[dev,accurate]
pip install flask

echo "Done. Run the sidecar using the virtual environment interpreter:"
echo "source .venv/bin/activate && python main.py"
echo "Or on Windows:"
echo ".\\.venv\\Scripts\\activate && python main.py"
