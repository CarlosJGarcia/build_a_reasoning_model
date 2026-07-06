# 2.5 Loading pretrained models
import torch

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
device = get_device()
print(f"device: {device}\n")

# Downloads Qwen3 0.6B into the "qwen" folder
download_qwen3_small(kind="base", tokenizer_only=False, out_dir="qwen3")
print("Quen3 0.6B Downloaded")