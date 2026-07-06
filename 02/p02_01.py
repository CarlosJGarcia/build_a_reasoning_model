import torch

print(f"\nPyTorch version {torch.__version__}")

if torch.cuda.is_available():
    print(f"CUDA GPU: {torch.cuda.get_device_name(0)}")

elif torch.xpu.is_available():
    print(f"Intel GPU: {torch.xpu.get_device_name(0)}")

elif torch.backends.mps.is_available():
    print("Apple Silicon GPU")

else:
    print("Only CPU")
print()