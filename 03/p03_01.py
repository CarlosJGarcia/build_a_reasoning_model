# Loading a pretrained model to generate text

import torch

torch.set_float32_matmul_precision('high')                        # PyTorch to use TF32 on RTX 5090 for massive speedups on torch.compile()
import torch._inductor.config as inductor_config                  # Tell PyTorch's inductor to stop warning you about the growing KV cache shapes
inductor_config.triton.cudagraph_dynamic_shape_warn_limit = None  # Silence a warning about the KV Cache growing during iteration loops

from pathlib import Path
from rich.console import Console
from rich.markdown import Markdown
from reasoning_from_scratch.ch02 import get_device
from reasoning_from_scratch.ch02 import generate_text_basic_stream_cache
from reasoning_from_scratch.qwen3 import download_qwen3_small, Qwen3Tokenizer, Qwen3Model, QWEN_CONFIG_06_B


def load_model_and_tokenizer(which_model, device, use_compile, local_dir="qwen3"):
    
    # Load the tokenizer
    if which_model == "base":
        # download_qwen3_small(kind="base", tokenizer_only=False, out_dir=local_dir)
        tokenizer_path = Path("../models/qwen3") / "tokenizer-base.json"
        model_path = Path("../models/qwen3") / "qwen3-0.6B-base.pth"
        tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)
        console.print(f"\nTokenizer created (from {tokenizer_path})", style="gold1", highlight=False)
    elif which_model == "reasoning":
        download_qwen3_small(kind="reasoning", tokenizer_only=False, out_dir=local_dir)
        tokenizer_path = Path(local_dir) / "tokenizer-reasoning.json"
        model_path = Path(local_dir) / "qwen3-0.6B-reasoning.pth"
        tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path, apply_chat_template=True, add_generation_prompt=True, add_thinking=True)
    else:
        raise ValueError(f"Invalid choice: which_model={which_model}")

    # Load the model, send to device(CPU/GPU) and compile (optional)
    model = Qwen3Model(QWEN_CONFIG_06_B)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)
    console.print(f"Qwen3 loaded in ({device}) GPU and VRAM (from {model_path})\n", style="gold1", highlight=False)
    if use_compile: 
        torch._dynamo.config.allow_unspec_int_on_nn_module = True
        model = torch.compile(model, mode="max-autotune-no-cudagraphs")

    return model, tokenizer


WHICH_MODEL = "base"  
MAX_NEW_TOKENS=2048

console = Console()
device = get_device()

model, tokenizer = load_model_and_tokenizer(which_model=WHICH_MODEL, device=device, use_compile=False)
# model, tokenizer = load_model_and_tokenizer(which_model=WHICH_MODEL, device=device, use_compile=True)

# Tokenizes the prompt and sends it to the GPU. The prompt uses LaTeX
prompt = r"If $a+b=3$ and $ab=\tfrac{13}{6}$, what is the value of $a^2+b^2$?" 
input_token_ids_tensor = torch.tensor(tokenizer.encode(prompt), device=device).unsqueeze(0)

all_token_ids = []

# Sends the prompt and gets the output stream from the LLM
token_stream = generate_text_basic_stream_cache(model=model, token_ids=input_token_ids_tensor, max_new_tokens=MAX_NEW_TOKENS, eos_token_id=tokenizer.eos_token_id)

# Decode the token stream to text
for token in token_stream:
    token_id = token.squeeze(0)
    decoded_id = tokenizer.decode(token_id.tolist())
    print(decoded_id, end="", flush=True)
    all_token_ids.append(token_id)
all_tokens = tokenizer.decode(all_token_ids)

# Muestra el resultado en formato legible
print(f"\nPrompt: {prompt}")
console.print(f"\nResponse: ", style="white")
console.print(Markdown(all_tokens), style="white")
print()