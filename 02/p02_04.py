# 2.6 Understanding the sequential LLM text generation process - how to use the LLM to generate a single token
# Generate a single token = predict what will be the next token given a sequence of tokens

import torch
from pathlib import Path
from p02_03 import get_device
from rich.console import Console
from reasoning_from_scratch.qwen3 import Qwen3Tokenizer, Qwen3Model, QWEN_CONFIG_06_B


console = Console()

# Load the tokenizer
tokenizer_path = Path("qwen3") / "tokenizer-base.json"
tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)
console.print(f"\nTokenizer created (from {tokenizer_path})", style="gold1")

# Load the model and send to GPU
device = get_device()
model_path = Path("qwen3") / "qwen3-0.6B-base.pth"
model = Qwen3Model(QWEN_CONFIG_06_B)
model.load_state_dict(torch.load(model_path))
model.to(device)
console.print(f"Qwen3 loaded in ({device}) GPU and VRAM (from {model_path})\n", style="gold1", highlight=False)


# Tokenize (encode)
prompt = "Explain large language models."
input_token_ids_list = tokenizer.encode(prompt)
print(f"Prompt: {prompt}")
print(f"Number of input tokens: {len(input_token_ids_list)}")


# Convert tokens to tensor and send to GPU
input_tensor = torch.tensor(input_token_ids_list)
input_tensor_fmt = input_tensor.unsqueeze(0)  
input_tensor_fmt = input_tensor_fmt.to(device)


# Inference
with torch.inference_mode():
    output_tensor = model(input_tensor_fmt)  
output_tensor_fmt = output_tensor.squeeze(0) 
print(f"Formatted output tensor shape: {output_tensor_fmt.shape}")

last_token = output_tensor_fmt[-1]
console.print(f"Last token: {tokenizer.decode([20286])}", style="white")
print()

