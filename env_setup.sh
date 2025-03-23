python -m venv .venv
source .venv/bin/activate
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -e .