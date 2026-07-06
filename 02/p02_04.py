import torch
from pathlib import Path
from rich.console import Console
from reasoning_from_scratch.qwen3 import Qwen3Tokenizer, Qwen3Model, QWEN_CONFIG_06_B

from p02_03 import get_device

# Crea el objeto tokenizer
tokenizer_path = Path("qwen3") / "tokenizer-base.json"
tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)

console = Console()
prompt = "Explain large language models."
input_token_ids_list = tokenizer.encode(prompt)
print(f"\nNumber of input tokens: {len(input_token_ids_list)}")


device = get_device()
input_tensor = torch.tensor(input_token_ids_list)
input_tensor_fmt = input_tensor.unsqueeze(0)  
input_tensor_fmt = input_tensor_fmt.to(device)

# Load the model
model_path = Path("qwen3") / "qwen3-0.6B-base.pth"
model = Qwen3Model(QWEN_CONFIG_06_B)
model.load_state_dict(torch.load(model_path))
model.to(device)
console.print("Qwen3 loaded in VRAM\n", style="gold1")


with torch.inference_mode():
    output_tensor = model(input_tensor_fmt)  
output_tensor_fmt = output_tensor.squeeze(0) 
print(f"Formatted Output tensor shape: {output_tensor_fmt.shape}")
print()
