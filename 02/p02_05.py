# 2.7 Coding a minimal text generation function. Instead of a single token, but based on the token-generation step, generate a whole text.

import torch
from pathlib import Path
from p02_03 import get_device
from rich.console import Console
from reasoning_from_scratch.qwen3 import Qwen3Tokenizer, Qwen3Model, QWEN_CONFIG_06_B

# Disable gradient tracking for speed and memory efficiency
@torch.inference_mode()                                                              
def generate_text_basic_stream(model, token_ids, max_new_tokens, eos_token_id=None):
    # Switch model to evaluation mode (best practice)
    model.eval()                                                                     

    for _ in range(max_new_tokens):
        out = model(token_ids)[:, -1]                          # Scores of the 'predicted next token'
        next_token = torch.argmax(out, dim=-1, keepdim=True)   # Selects 'predicted next token' based on the scores

        """
        if (eos_token_id is not None                           
                and torch.all(next_token == eos_token_id)):
            break
        """
        if (eos_token_id is not None and next_token.item() == eos_token_id):
            break

        yield next_token                                       # Yield the predicted token as soon as it’s generated.

        token_ids = torch.cat([token_ids, next_token], dim=1)  # Append the predicted token to the sequence.

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


# Tokenizes the prompt and sends it to the GPU
reply = ""
max_new_tokens = 100                    
eot_token_id = tokenizer.encode("<|endoftext|>")[0]
prompt = "Explain large language models in a single sentence."
input_token_ids_tensor = torch.tensor(tokenizer.encode(prompt), device=device).unsqueeze(0)

# Sends the prompt and gets the output stream from the LLM
token_stream = generate_text_basic_stream(model=model, token_ids=input_token_ids_tensor, max_new_tokens=max_new_tokens, eos_token_id=eot_token_id)

# Decode the token stream to text
for token in token_stream:
    token_id = token.squeeze(0).tolist()     
    token_text = tokenizer.decode(token_id)
    reply += token_text

print(f"Prompt: {prompt}")
console.print(f"Response: {reply}\n", style="white", highlight=False)

