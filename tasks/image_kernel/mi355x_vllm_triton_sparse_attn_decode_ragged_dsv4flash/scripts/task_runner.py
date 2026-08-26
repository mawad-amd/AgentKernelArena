#!/usr/bin/env python3
"""Harness for the vLLM Triton DeepSeek-V4-Flash sparse-attention decode pair
``_sparse_attn_decode_partial_kernel`` + ``_sparse_attn_decode_reduce_kernel``
(rocm_aiter_mla_sparse.py).

Both kernels are scored together through the ``_rocm_sparse_attn_decode_ragged_triton``
entry point, so moving work between the split and reduce stages is a legal
optimisation as long as the pair stays correct.

The kernel module is loaded from the editable workspace copy of the in-image source
tree so an optimizing agent's edits take effect (Triton JIT recompiles on source change).

The fp8_ds_mla cache pack/unpack logic (``_pack_fp8_ds_mla_cache``, ``_read_cache_rows``)
follows vLLM's own test for this kernel,
``tests/kernels/attention/test_rocm_triton_attn_dsv4.py`` (vLLM, Apache-2.0), so the bytes
this harness feeds the kernel match what vLLM's production store path writes.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
SPEC = json.loads((WORKSPACE / "session_cases.json").read_text())
OPERATOR = SPEC["operator"]
CASES = SPEC["cases"]

assert OPERATOR == "sparse_attn_decode_ragged", (
    f"task_runner is specific to sparse_attn_decode_ragged, got {OPERATOR!r}"
)

REPO_SUBDIR = "vllm_v1_attention_ops"
KERNEL_FILE = "rocm_aiter_mla_sparse.py"
EDIT_MODULE_NAME = "vllm.v1.attention.ops._ka_rocm_aiter_mla_sparse"

# fp8_ds_mla packed cache layout: 576 bytes of token payload
# (448 NoPE fp8 + 64 RoPE bf16) followed by 8 bytes of e8m0 group scales per token.
_TOKEN_BYTES = 576
_SCALE_BYTES = 8
_ROW_BYTES = _TOKEN_BYTES + _SCALE_BYTES  # 584
_NOPE_GROUP = 64  # one e8m0 scale per 64 NoPE elements -> 7 scales


def _configure() -> None:
    for key in ("GPU_ARCHS", "PYTORCH_ROCM_ARCH", "AMDGPU_TARGETS", "GPU_TARGETS"):
        os.environ.setdefault(key, "gfx950")
    # The injected benchmark helpers do `from _aka_benchmark import ...`, which only
    # resolves when scripts/ happens to be on sys.path. It is when this file is run
    # as `python3 scripts/task_runner.py`, but forge_driver loads this module by
    # file path from the workspace root, where it is not — so bench-mode fails with
    # a misleading `No module named 'src'` from the fallback branch. Put scripts/
    # on the path explicitly so the entry point does not decide whether timing works.
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    os.chdir(WORKSPACE)


# >>> AKA-GENERATED: shared CUDA-graph benchmark helpers - edit src/tools/perf/vllm_cuda_graph_block.py then run `make sync-perf-helpers` >>>
def _measure_cuda_event_fallback(*args, **kwargs):
    raise RuntimeError(
        "CUDA-graph benchmark helpers were not materialized. "
        "Run this task through AgentKernelArena so setup_workspace() can inject "
        "src/tools/perf/vllm_cuda_graph_block.py into the workspace."
    )


def _benchmark_cuda_graph_or_events(*args, **kwargs):
    raise RuntimeError(
        "CUDA-graph benchmark helpers were not materialized. "
        "Run this task through AgentKernelArena so setup_workspace() can inject "
        "src/tools/perf/vllm_cuda_graph_block.py into the workspace."
    )
# <<< AKA-GENERATED <<<


def _write_report(rows: list[dict]) -> None:
    report_dir = WORKSPACE / "build"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "performance_report.json").write_text(json.dumps(rows, indent=2))


def _torch():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("ROCm GPU is required")
    return torch


def _use_fnuz() -> bool:
    from vllm.platforms import current_platform

    return bool(current_platform.is_fp8_fnuz())


def _load_kernel_module():
    # rocm_aiter_mla_sparse.py uses only absolute imports and registers no custom
    # ops at import time, so a straight file-path load of the edited workspace copy
    # is sufficient for the agent's edits to take effect.
    import vllm  # noqa: F401  (ensure platform init)

    path = WORKSPACE / REPO_SUBDIR / KERNEL_FILE
    spec = importlib.util.spec_from_file_location(EDIT_MODULE_NAME, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[EDIT_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def _pack_fp8_ds_mla_cache(kv, block_size: int, use_fnuz: bool):
    """Quantise a dense bf16 latent into the fp8_ds_mla paged layout.

    Uses vLLM's own store path rather than reimplementing the packing, so the
    cache the kernel reads is byte-identical to what production writes.
    """
    torch = _torch()
    from vllm.models.deepseek_v4.common.ops.cache_utils import (
        quantize_and_insert_k_cache,
    )

    num_tokens = kv.shape[0]
    num_blocks = (num_tokens + block_size - 1) // block_size
    cache = torch.zeros(
        (num_blocks, block_size, _ROW_BYTES), dtype=torch.uint8, device=kv.device
    )
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=kv.device)
    quantize_and_insert_k_cache(
        kv, cache, slot_mapping, block_size=block_size, use_fnuz=use_fnuz
    )
    return cache


def _read_cache_rows(cache, slots, block_size: int, use_fnuz: bool):
    """Dequantise selected cache rows back to float32 [n, head_dim].

    Mirrors the packed layout: 448 fp8 NoPE scaled by 7 e8m0 group exponents,
    then 64 bf16 RoPE.
    """
    torch = _torch()
    flat = cache.view(torch.uint8).flatten()
    block_idx = slots // block_size
    pos = slots % block_size
    block_base = block_idx * cache.stride(0)
    token_base = block_base + pos * _TOKEN_BYTES
    scale_base = block_base + block_size * _TOKEN_BYTES + pos * _SCALE_BYTES

    fp8_dtype = torch.float8_e4m3fnuz if use_fnuz else torch.float8_e4m3fn
    nope_dim = CASES[0]["params"]["nope_head_dim"]
    rope_dim = CASES[0]["params"]["rope_head_dim"]
    n_scales = nope_dim // _NOPE_GROUP

    nope_off = torch.arange(nope_dim, device=cache.device)
    nope = flat[token_base[:, None] + nope_off].view(fp8_dtype).to(torch.float32)
    scale_off = torch.arange(n_scales, device=cache.device)
    scales = torch.exp2(
        flat[scale_base[:, None] + scale_off].to(torch.float32) - 127.0
    )
    nope = nope * scales.repeat_interleave(_NOPE_GROUP, dim=1)

    rope_off = torch.arange(rope_dim * 2, device=cache.device)
    rope = (
        flat[token_base[:, None] + nope_dim + rope_off]
        .contiguous()
        .view(torch.bfloat16)
        .to(torch.float32)
    )
    return torch.cat([nope, rope], dim=1)


def _make(case: dict) -> dict:
    """Build a case at its scored shape.

    No correctness/performance switch: the shape that is timed is the shape that
    is validated, or the scored path can differ from the checked one.
    """
    torch = _torch()
    p = dict(case["params"])
    num_queries = p["num_queries"]
    num_heads = p["num_heads"]
    nope_head_dim = p["nope_head_dim"]
    rope_head_dim = p["rope_head_dim"]
    head_dim = nope_head_dim + rope_head_dim
    num_kv = p["num_kv"]
    topk = min(p["topk"], num_kv)
    block_size = p["block_size"]
    use_fnuz = _use_fnuz()

    torch.manual_seed(31)
    # Unit variance on both sides is derived from the model, not chosen.
    # q: DeepSeek applies a WEIGHTLESS per-head RMS norm immediately before
    #    attention -- `q *= rsqrt(q.square().mean(-1) + eps)` in the reference
    #    inference/model.py, and vLLM documents the same at
    #    deepseek_v4/attention.py:634 ("per-head RMSNorm (no weight)"). So every
    #    q head-vector has RMS exactly 1 by construction: sigma_q = 1.
    # kv: passes through kv_norm, an RMSNorm with an O(1) learned weight, so
    #     sigma_kv is order 1. Magnitude is safe for fp8 because the cache stores
    #     a per-64-element e8m0 scale, which absorbs the range.
    # Score std is sigma_q * sigma_kv * sqrt(D) * scale, and sqrt(D)*scale == 1,
    # so trained attention here sits at score std ~1. A degenerate 0.125/0.125
    # would give 0.016 -- 512 near-identical logits, a flat softmax a kernel could
    # exploit by skipping the max subtraction.
    q = torch.randn(
        (num_queries, num_heads, head_dim), device="cuda", dtype=torch.bfloat16
    )
    kv = torch.randn((num_kv, head_dim), device="cuda", dtype=torch.bfloat16)
    cache = _pack_fp8_ds_mla_cache(kv, block_size, use_fnuz)

    # Ragged CSR selection: each query attends to `topk` distinct cache slots.
    rows = []
    for qi in range(num_queries):
        perm = torch.randperm(num_kv, device="cuda")[:topk]
        rows.append(perm.sort().values)
    indices = torch.cat(rows).to(torch.int32).contiguous()
    indptr = torch.arange(
        0, (num_queries + 1) * topk, topk, device="cuda", dtype=torch.int32
    )

    # Per-head learnable attention sink. DeepSeek-V4 ships one per layer (44 in the
    # V4-Flash checkpoint) and vLLM passes it on every decode call, so None would
    # leave the kernel's sink branch unexercised and time the wrong shape.
    # Magnitudes taken from the real V4-Flash checkpoint: per-layer attn_sink is
    # 64 values with mean ~0.35-0.62 and std ~0.2-0.64 (layers 0-3 sampled).
    attn_sink = (
        torch.randn((num_heads,), device="cuda", dtype=torch.float32) * 0.5 + 0.5
    )

    return {
        "cfg": case,
        "module": _load_kernel_module(),
        "q": q,
        "attn_sink": attn_sink,
        "cache": cache,
        "indices": indices,
        "indptr": indptr,
        "rows": rows,
        "scale": head_dim**-0.5,
        "nope_head_dim": nope_head_dim,
        "rope_head_dim": rope_head_dim,
        "block_size": block_size,
        "use_fnuz": use_fnuz,
    }


def _run(inputs: dict):
    return inputs["module"]._rocm_sparse_attn_decode_ragged_triton(
        q=inputs["q"],
        main_cache=inputs["cache"],
        main_indices=inputs["indices"],
        main_indptr=inputs["indptr"],
        scale=inputs["scale"],
        attn_sink=inputs["attn_sink"],
        nope_head_dim=inputs["nope_head_dim"],
        rope_head_dim=inputs["rope_head_dim"],
    )


def _reference(inputs: dict):
    """Dequantise the selected rows and attend densely, in float32, with the sink.

    The latent is both K and V, so this is a single gather followed by
    softmax(q.K^T * scale) @ K.
    """
    torch = _torch()
    q = inputs["q"].float()
    out = torch.empty_like(q)
    for qi in range(q.shape[0]):
        kv = _read_cache_rows(
            inputs["cache"], inputs["rows"][qi], inputs["block_size"], inputs["use_fnuz"]
        )
        scores = torch.einsum("hd,kd->hk", q[qi], kv) * inputs["scale"]
        # The sink is an extra logit whose value vector is zero: it takes mass out
        # of the softmax denominator but contributes nothing to the output.
        sink = inputs["attn_sink"].float().reshape(-1, 1)
        probs = torch.softmax(torch.cat([scores, sink], dim=-1), dim=-1)[:, :-1]
        out[qi] = torch.einsum("hk,kd->hd", probs, kv)
    return out.to(inputs["q"].dtype)


def _assert_close(inputs: dict, got) -> None:
    _torch().testing.assert_close(got, _reference(inputs), atol=2e-2, rtol=2e-2)


def _perturb_inputs(inputs: dict) -> None:
    """Refresh data inputs in place with values no earlier launch has seen.

    A replayed graph reads the captured addresses, so writing through them
    changes what the scored kernel consumes. The CSR selection and the packed
    cache are structure, not data, so they stay fixed.
    """
    torch = _torch()
    torch.manual_seed(53)
    inputs["q"].normal_()


def _compile_smoke_case(case: dict) -> dict:
    """Shrink a case so the compile smoke test stays cheap. compile only."""
    smoke = {**case, "params": dict(case["params"])}
    smoke["params"]["num_kv"] = min(case["params"]["num_kv"], 128)
    smoke["params"]["topk"] = min(case["params"]["topk"], 64)
    smoke["params"]["num_heads"] = min(case["params"]["num_heads"], 8)
    return smoke


def _assert_timed_outputs(inputs: dict, timed) -> None:
    """Validate the invocation the benchmark actually timed."""
    if not timed.bound:
        raise RuntimeError("benchmark did not expose the timed invocation")
    _perturb_inputs(inputs)
    if timed.outputs is not None:
        timed.outputs.fill_(float("nan"))
    _assert_close(inputs, timed.rerun())


def run_compile() -> None:
    inputs = _make(_compile_smoke_case(CASES[0]))
    _run(inputs)
    _torch().cuda.synchronize()
    print(f"{OPERATOR} compile smoke: PASS")


def run_correctness() -> None:
    torch = _torch()
    for case in CASES:
        inputs = _make(case)
        got = _run(inputs)
        torch.cuda.synchronize()
        _assert_close(inputs, got)
        print("correctness PASS", case["id"])


def run_performance() -> None:
    rows = []
    for case in CASES:
        inputs = _make(case)
        _run(inputs)
        _torch().cuda.synchronize()
        timed = _TimedRun()
        execution_time_ms, bench_meta = _benchmark_cuda_graph_or_events(
            lambda: _run(inputs),
            warmup=10,
            repetition=100,
            target_ms=1.0,
            max_graph_repeats=1000,
            timed_run=timed,
        )
        _assert_timed_outputs(inputs, timed)
        metadata = {
            **case["params"],
            "model": case.get("model"),
            "layer_class": case.get("layer_class"),
            "observed_num_splits": case.get("observed_num_splits"),
            "decode_gpu_share_eager": case.get("decode_gpu_share_eager"),
            "denominator": "sum of GPU time over decode-phase kernels, eager",
            "benchmark_method": bench_meta.get("benchmark_method"),
        }
        metadata.update(
            {k: v for k, v in bench_meta.items() if k.startswith("benchmark_")}
        )
        rows.append(
            {
                "test_case_id": case["id"],
                "shape": case.get("trace_input_shapes"),
                "execution_time_ms": execution_time_ms,
                "metadata": metadata,
            }
        )
        print(
            case["id"],
            f"{execution_time_ms:.6f} ms",
            bench_meta.get("benchmark_method"),
            bench_meta.get("benchmark_fallback_reason", ""),
        )
    _write_report(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=["compile", "correctness", "performance", "manifest"]
    )
    mode = parser.parse_args().mode
    if mode == "manifest":
        print(json.dumps(SPEC, indent=2))
        return
    _configure()
    if mode == "compile":
        run_compile()
    elif mode == "correctness":
        run_correctness()
    else:
        run_performance()


if __name__ == "__main__":
    main()
