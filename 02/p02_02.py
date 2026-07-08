# 2.4 Preparing input texts for LLMs

from pathlib import Path
from rich.console import Console
from reasoning_from_scratch.qwen3 import Qwen3Tokenizer


# Creates "qwen" folder and downloads "tokenizer-base.json" 
#download_qwen3_small(kind="base", tokenizer_only=True, out_dir="qwen3")

console = Console()

# Crea los objetos tokenizer_path y tokenizer. En Python el operador / se traduce automáticamente por el "path separator" de Linux, macOS o Windows.
tokenizer_path = Path("qwen3") / "tokenizer-base.json"
tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)
console.print(f"\nTokenizer created (from {tokenizer_path})\n", style="gold1")


# Test the tokernizer: Encode
prompt = "The quick brown fox jumps over the lazy dog. Milana bonita."
input_token_ids_list = tokenizer.encode(prompt)
text = tokenizer.decode(input_token_ids_list)
print(f"Input: {text}")
print("Token IDs:")
for i in input_token_ids_list:
    print(f"{i} --> {tokenizer.decode([i])}")
print()

# Test the tokernizer: Round trip Encode + Decode
prompt = "Explain large language models."
input_token_ids_list = tokenizer.encode(prompt)
text = tokenizer.decode(input_token_ids_list)
print(f"Input: {prompt}")
console.print(f"Output: {text}\n", style="white")