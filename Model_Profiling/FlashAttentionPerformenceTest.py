import torch
import torch.nn.functional as F
import time
import time
from torch.profiler import profile, record_function, ProfilerActivity
import subprocess
import datetime

curr_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
from torch.nn.attention import sdpa_kernel, SDPBackend


import math

# Efficient implementation equivalent to the following:
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value

def test_sdpa(backend, enable_gqa=False):
    print(f"Testing {backend}...")
    query = torch.rand(32, 8, 512, 128, dtype=torch.float16, device="cuda")
    key = torch.rand(32, 8, 512, 128, dtype=torch.float16, device="cuda")
    value = torch.rand(32, 8, 512, 128, dtype=torch.float16, device="cuda")
    # mask = torch.rand(32, 8, 512, 512, dtype=torch.float16, device="cuda")
    mask = None
    
    if enable_gqa:  # Test with GQA
        query = torch.rand(32, 32, 512, 128, dtype=torch.float16, device="cuda")
        key = torch.rand(32, 8, 512, 128, dtype=torch.float16, device="cuda")
        value = torch.rand(32, 8, 512, 128, dtype=torch.float16, device="cuda")
    
    initial = 40
    final = 40
    losses = []
    execution_time = []

    ab = time.time()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f'/lus/eagle/projects/cms_l1t_fm/quantization/profile_runs/logs/{curr_time}/{backend}'),
        record_shapes=True,
        profile_memory=True
    ) as prof:
        for i in range(200):
            with torch.no_grad():
                start = time.time()
                # output = scaled_dot_product_attention(query, key, value, mask, enable_gqa=enable_gqa) # Does not give any benefit
                # with sdpa_kernel(backends=[backend]):
                #     output = F.scaled_dot_product_attention(query, key, value, mask, enable_gqa=enable_gqa)
                end = time.time()

                execution_time.append(end-start)
            prof.step()
            
    ba = time.time()
    print('Total Time', ba-ab)

    print('test execution time', sum(execution_time[initial:])/len(execution_time[initial:]))
    print(execution_time[final:])
    
    print("Output shape:", output.shape, "\n")

if __name__ == "__main__":
    if torch.cuda.is_available():
        test_sdpa(SDPBackend.MATH)
        #test_sdpa(SDPBackend.MATH, enable_gqa=True)  # Test GQA case
        test_sdpa(SDPBackend.EFFICIENT_ATTENTION)
        test_sdpa(SDPBackend.FLASH_ATTENTION)
    else:
        print("CUDA is not available. Please run on a GPU.")


