# Faster inference via KV Cache. KV = Keys and Values in the attention mechanism

import torch
from reasoning_from_scratch.qwen3 import KVCache

@torch.inference_mode()
def generate_text_basic_stream_cache(model, token_ids, max_new_tokens, eos_token_id=None):

    model.eval()
    cache = KVCache(n_layers=model.cfg["n_layers"])  
    model.reset_kv_cache()                            

    out = model(token_ids, cache=cache)[:, -1]       
    for _ in range(max_new_tokens):
        next_token = torch.argmax(out, dim=-1, keepdim=True)

        if (eos_token_id is not None and torch.all(next_token == eos_token_id)):
            break

        yield next_token
        out = model(next_token, cache=cache)[:, -1]  #3

print("Todo bien")
