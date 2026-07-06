**Instalación del entorno** \
conda create -n reasoning_scratch_cuda python=3.11.15 \
pip install "torch>=2.10.0" torchvision torchaudio "tokenizers>=0.22.2" reasoning-from-scratch   --extra-index-url https://download.pytorch.org/whl/cu128   --trusted-host pypi.org   --trusted-host files.pythonhosted.org   --trusted-host download.pytorch.org   --trusted-host download-r2.pytorch.org   --trusted-host pypi.nvidia.com \


**Importante:** \
No hacer `$ conda update --all -c conda-forge -y` porque instala una versión de PyTorch sin CUDA \

Para comprobar que está funcionando CUDA en PyTorch ejecutar `/appendix_a/version.py`
