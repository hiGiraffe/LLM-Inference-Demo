import random
from typing import List, Optional, Tuple

import torch
from time import perf_counter
from allclose_default import get_default_atol, get_default_rtol

from vllm import _custom_ops as ops
from vllm.utils import get_max_shared_memory_bytes, is_hip, create_kv_caches_with_random
import timeit

FLOAT32_BYTES = torch.finfo(torch.float).bits // 8
# This will change depending on the compute capability.
# - 512 as a buffer
# 改动测试1
MAX_SEQ_LEN = 4096

# There may not be enough gpu memory due to large NUM_BLOCKS.
# Reduce NUM_BLOCKS when it happens.
NUM_BLOCKS = 4321  # Arbitrary values for testing
PARTITION_SIZE = 512 # V2才考虑
# flshattF and tritonflashattF supported: {torch.float16, torch.bfloat16}
DTYPES = [torch.float]
NUM_GEN_SEQS = [2,4,8,16,32,64,128]  # Arbitrary values for testing
NUM_PREFILL_SEQS = [3]  # Arbitrary values for testing
NUM_HEADS = [(40, 40), (64, 8)]  # Arbitrary values for testing

# FlashAttention forward only supports head dimension at most 128
# https://github.com/ROCmSoftwarePlatform/flash-attention/blob/3d2b6f5d037782cc2c906909a46fb7e2e1b48b25/csrc/flash_attn_rocm/flash_api.cpp#L62
HEAD_SIZES = [64, 80, 96, 112, 128, 256
              ] if not is_hip() else [64, 80, 96, 112, 128]

BLOCK_SIZES = [16]
USE_ALIBI = [False]
KV_CACHE_DTYPE = ["auto"]
SEEDS = [0]
CUDA_DEVICES = ["cuda:0"]




def ref_masked_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale: float,
        attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    attn_weights = scale * torch.einsum("qhd,khd->hqk", query, key).float()
    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask.float()
    attn_weights = torch.softmax(attn_weights, dim=-1).to(value.dtype)
    out = torch.einsum("hqk,khd->qhd", attn_weights, value)
    return out


def ref_single_query_cached_kv_attention(
        output: torch.Tensor,
        query: torch.Tensor,
        num_queries_per_kv: int,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
        scale: float,
        alibi_slopes: Optional[torch.Tensor],
) -> None:
    num_query_heads = query.shape[1]
    num_kv_heads = value_cache.shape[1]
    head_size = value_cache.shape[2]
    block_size = value_cache.shape[3]
    num_seqs = query.shape[0]

    block_tables = block_tables.cpu().tolist()
    context_lens = context_lens.cpu().tolist()
    for i in range(num_seqs):
        q = query[i].unsqueeze(0)
        block_table = block_tables[i]
        context_len = int(context_lens[i])

        keys = []
        values = []
        for j in range(context_len):
            block_number = int(block_table[j // block_size])
            block_offset = j % block_size

            k = key_cache[block_number, :, :, block_offset, :]
            k = k.reshape(num_kv_heads, head_size)
            keys.append(k)

            v = value_cache[block_number, :, :, block_offset]
            values.append(v)
        keys = torch.stack(keys, dim=0)
        values = torch.stack(values, dim=0)
        if num_queries_per_kv > 1:
            # Handle MQA and GQA
            keys = torch.repeat_interleave(keys, num_queries_per_kv, dim=1)
            values = torch.repeat_interleave(values, num_queries_per_kv, dim=1)

        alibi_bias = None
        if alibi_slopes is not None:
            # Create the ALiBi bias used in the paged attention kernel.
            position_ids = torch.arange(context_len).int()
            alibi_bias = (position_ids - context_len + 1).float()
            alibi_bias = alibi_slopes.view(-1, 1, 1) * alibi_bias.view(
                1, 1, -1)

        out = ref_masked_attention(q, keys, values, scale, alibi_bias)
        out = out.view(num_query_heads, head_size)
        output[i].copy_(out, non_blocking=True)


# @pytest.mark.parametrize("version", ["v1"])
# @pytest.mark.parametrize("num_seqs", NUM_GEN_SEQS)
# @pytest.mark.parametrize("num_heads", NUM_HEADS)
# @pytest.mark.parametrize("head_size", HEAD_SIZES)
# @pytest.mark.parametrize("use_alibi", USE_ALIBI)
# @pytest.mark.parametrize("block_size", BLOCK_SIZES)
# @pytest.mark.parametrize("dtype", DTYPES)
# @pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPE)
# @pytest.mark.parametrize("seed", SEEDS)
# @pytest.mark.parametrize("device", CUDA_DEVICES)
def test_paged_attention(
        kv_cache_factory,
        version: str,
        num_seqs: int,
        num_heads: Tuple[int, int],
        head_size: int,
        use_alibi: bool,
        block_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: str,
        seed: int,
        device: str,
) -> None:
    random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_default_device(device)
    scale = float(1.0 / (head_size ** 0.5))
    num_query_heads, num_kv_heads = num_heads
    query = torch.empty(num_seqs, num_query_heads, head_size, dtype=dtype)
    query.uniform_(-scale, scale)

    assert num_query_heads % num_kv_heads == 0
    num_queries_per_kv = num_query_heads // num_kv_heads
    alibi_slopes = None
    if use_alibi:
        alibi_slopes = torch.randn(num_query_heads, dtype=torch.float)

    context_lens = [MAX_SEQ_LEN for _ in range(num_seqs)]
    context_lens[-1] = MAX_SEQ_LEN
    max_context_len = max(context_lens)
    context_lens = torch.tensor(context_lens, dtype=torch.int)

    # Create the block tables.
    max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size
    block_tables = []
    for _ in range(num_seqs):
        block_table = [
            random.randint(0, NUM_BLOCKS - 1)
            for _ in range(max_num_blocks_per_seq)
        ]
        block_tables.append(block_table)
    block_tables = torch.tensor(block_tables, dtype=torch.int)

    # Create the KV caches.
    key_caches, value_caches = kv_cache_factory(NUM_BLOCKS, block_size, 1,
                                                num_kv_heads, head_size,
                                                kv_cache_dtype, dtype, seed,
                                                device)
    key_cache, value_cache = key_caches[0], value_caches[0]

    # Using default kv_scale
    kv_scale = 1.0

    # Call the paged attention kernel.
    output = torch.empty_like(query)

    # 输出物理位置
    #print("output is on ", output.device)
    #print("query is on ", query.device)
    #print("key_cache is on ", key_cache.device)
    #print("value_cache is on ", value_cache.device)

    # 代码开始时间
    start_time = timeit.default_timer()
    # start_time = perf_counter()
    if version == "v1":
        ops.paged_attention_v1(
            output,
            query,
            key_cache,
            value_cache,
            num_kv_heads,
            scale,
            block_tables,
            context_lens,
            block_size,
            max_context_len,
            alibi_slopes,
            kv_cache_dtype,
            kv_scale,
        )
    elif version == "v2":
        num_partitions = ((max_context_len + PARTITION_SIZE - 1) //
                          PARTITION_SIZE)
        assert PARTITION_SIZE % block_size == 0
        num_seqs, num_heads, head_size = output.shape
        tmp_output = torch.empty(
            size=(num_seqs, num_heads, num_partitions, head_size),
            dtype=output.dtype,
        )
        exp_sums = torch.empty(
            size=(num_seqs, num_heads, num_partitions),
            dtype=torch.float32,
        )
        max_logits = torch.empty_like(exp_sums)
        ops.paged_attention_v2(
            output,
            exp_sums,
            max_logits,
            tmp_output,
            query,
            key_cache,
            value_cache,
            num_kv_heads,
            scale,
            block_tables,
            context_lens,
            block_size,
            max_context_len,
            alibi_slopes,
            kv_cache_dtype,
            kv_scale,
        )
    else:
        raise AssertionError(f"Unknown version: {version}")

    # 执行时间
    torch.cuda.synchronize(device)

    # end_time = perf_counter()
    # elapsed_time.append(end_time - start_time)
    elapsed_time = timeit.default_timer() - start_time
    print("num seqs = ",num_seqs)
    print("block size = ", block_size)
    print("head size = ",head_size)
    print("num heads = ",num_heads)
    print("elapsed_time = ", elapsed_time)
    # Run the reference implementation.
    # if kv_cache_dtype == "fp8":
    #     # Convert cache data back to dtype.
    #     x = 16 // torch.tensor([], dtype=dtype).element_size()
    #     key_cache_shape = (NUM_BLOCKS, num_kv_heads, head_size // x,
    #                        block_size, x)
    #     dequantized_key_cache = torch.empty(size=key_cache_shape,
    #                                         dtype=dtype,
    #                                         device=device)
    #     ops.convert_fp8(key_cache, dequantized_key_cache)
    #     key_cache = dequantized_key_cache
    #
    #     value_cache_shape = value_cache.shape
    #     dequantized_value_cache = torch.empty(size=value_cache_shape,
    #                                           dtype=dtype,
    #                                           device=device)
    #     ops.convert_fp8(value_cache, dequantized_value_cache)
    #     value_cache = dequantized_value_cache

    ref_output = torch.empty_like(query)
    ref_single_query_cached_kv_attention(
        ref_output,
        query,
        num_queries_per_kv,
        key_cache,
        value_cache,
        block_tables,
        context_lens,
        scale,
        alibi_slopes,
    )

    # NOTE(woosuk): Due to the kernel-level differences in the two
    # implementations, there is a small numerical difference in the two
    # outputs. Thus, we use a relaxed tolerance for the test.
    atol = get_default_atol(output) if is_hip() else 1e-3
    rtol = get_default_rtol(output) if is_hip() else 1e-5

    # NOTE(zhaoyang): FP8 KV Cache will introduce quantization error,
    # so we use a relaxed tolerance for the test.
    atol, rtol = 1e-3, 1e-5
    if kv_cache_dtype == "fp8":
        atol, rtol = 1e-2, 1e-5
    assert torch.allclose(output, ref_output, atol=atol, rtol=rtol)



if __name__ == '__main__':
    version = "v1"
    #num_seqs = NUM_GEN_SEQS[0]
    use_alibi = USE_ALIBI[0]
    dtype = DTYPES[0]
    kv_cache_dtype = KV_CACHE_DTYPE[0]
    seed = SEEDS[0]
    device = CUDA_DEVICES[0]
    # num_heads = NUM_HEADS[0]
    # block_size = BLOCK_SIZES[0]
    # head_size = HEAD_SIZES[0]
    test_paged_attention(create_kv_caches_with_random, version, NUM_GEN_SEQS[0],
                         NUM_HEADS[0], HEAD_SIZES[0], use_alibi, BLOCK_SIZES[0],
                         dtype, kv_cache_dtype, seed, device)
    for num_seqs in NUM_GEN_SEQS:
        for num_heads in NUM_HEADS:
            for block_size in BLOCK_SIZES:
                for head_size in HEAD_SIZES:
                    test_paged_attention(create_kv_caches_with_random, version, num_seqs,
                             num_heads, head_size, use_alibi, block_size,
                             dtype, kv_cache_dtype, seed, device)