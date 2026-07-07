import torch

print("\nSoftware and hardware evaluation:")

# Software - PyTorch version
print(f"- PyTorch version: {torch.__version__}")

# Hardware - Each kind (CUDA, Apple or Intel) has a different eval. function that returns True or False
if torch.cuda.is_available():
    print(f"- CUDA GPU: {torch.cuda.get_device_name(0)}")

elif torch.xpu.is_available():
    print(f"- Intel XPU: {torch.xpu.get_device_name(0)}")

elif torch.backends.mps.is_available():
    print("- Apple Silicon MPS SoC")

else:
    print("Only CPU")
print()