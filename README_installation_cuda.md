create new venv
run: `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126`
then run: `pip install ipykernel`

Torch is installed with GPU support.

In `setup.py`, remove `torch`
Run setup.py installation

It works!