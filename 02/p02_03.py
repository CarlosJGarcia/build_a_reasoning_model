# 2.5 Loading pretrained models
import torch
from pathlib import Path
from rich.console import Console
from reasoning_from_scratch.qwen3 import download_qwen3_small, Qwen3Model, QWEN_CONFIG_06_B


def get_device(enable_tensor_cores=True):
    if torch.cuda.is_available():
        # Enable GPU
        device = torch.device("cuda")
        print("Using NVIDIA CUDA GPU")

        if enable_tensor_cores:
            # Enable Tensor Cores for faster matrix multiplications
            major, minor = map(int, torch.__version__.split(".")[:2])
            if (major, minor) >= (2, 9):
                torch.backends.cuda.matmul.fp32_precision = "tf32"
                torch.backends.cudnn.conv.fp32_precision = "tf32"
            else:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")

    elif torch.xpu.is_available():
        device = torch.device("xpu")
        print("Using Intel GPU")

    else:
        # Enable CPU. This can be used as well to bypass the GPU and force using the CPU.
        device = torch.device("cpu")
        print("Using CPU")

    return device


print()
console = Console()
device = get_device()
console.print(f"device: {device}\n", style="gold1")

# Downloads Qwen3 0.6B into the "qwen" folder. The name of the file is "qwen3-0.6B-base.pth" (no funciona por el proxy)
# download_qwen3_small(kind="base", tokenizer_only=False, out_dir="qwen3")
console.print("Qwen3 0.6B Downloaded", style="gold1", highlight=False)

# Load the model
model_path = Path("qwen3") / "qwen3-0.6B-base.pth"
model = Qwen3Model(QWEN_CONFIG_06_B)
model.load_state_dict(torch.load(model_path))
model.to(device)
console.print("Qwen3 loaded in VRAM\n", style="gold1")

# Shows model architecture
console.print("Model architecture:", style="gold1")
print(model)
print()

# Pause 0
# key = input("Press ENTER to exit.")
# print()  