from pathlib import Path
from reasoning_from_scratch.qwen3 import Qwen3Tokenizer, download_qwen3_small


# Creates "qwen" folder and downloads "tokenizer-base.json" 
download_qwen3_small(kind="base", tokenizer_only=True, out_dir="qwen3")

# Crea el objeto tokenizer
tokenizer_path = Path("qwen3") / "tokenizer-base.json"
tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)

# Test the tokernizer: Encode
prompt = "The quick brown fox jumps over the lazy dog"
input_token_ids_list = tokenizer.encode(prompt)
text = tokenizer.decode(input_token_ids_list)
print()
print(text)
print()
for i in input_token_ids_list:
    print(f"{i} --> {tokenizer.decode([i])}")
print()