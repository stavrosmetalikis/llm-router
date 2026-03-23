#!/bin/bash
cd "$(dirname "$0")"
git clone https://github.com/open-compress/claw-compactor.git
pip install -e claw-compactor[dev,accurate] --break-system-packages
pip install flask --break-system-packages
echo "Done. Run with: python3 main.py"
