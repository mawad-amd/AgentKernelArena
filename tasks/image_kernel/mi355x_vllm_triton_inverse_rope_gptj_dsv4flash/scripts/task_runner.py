#!/usr/bin/env python3
"""Harness for the vLLM Triton fused inverse GPT-J RoPE kernel
``_inverse_rope_gptj_kernel`` (rocm_aiter_mla_sparse.py).

Loaded from the editable workspace copy of the in-image source tree so an
optimizing agent's edits take effect (Triton JIT recompiles on source change).
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

assert OPERATOR == "inverse_rope_gptj", (
    f"task_runner is specific to inverse_rope_gptj, got {OPERATOR!r}"
)

REPO_SUBDIR = "vllm_v1_attention_ops"
KERNEL_FILE = "rocm_aiter_mla_sparse.py"
EDIT_MODULE_NAME = "vllm.v1.attention.ops._ka_rocm_aiter_mla_sparse"


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


def _load_kernel_module():
    import vllm  # noqa: F401  (ensure platform init)

    path = WORKSPACE / REPO_SUBDIR / KERNEL_FILE
    spec = importlib.util.spec_from_file_location(EDIT_MODULE_NAME, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[EDIT_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def _make(case: dict) -> dict:
    """Build a case at its scored shape.

    No correctness/performance switch: the timed shape is the validated shape.
    """
    torch = _torch()
    p = dict(case["params"])
    num_tokens = p["num_tokens"]
    num_heads = p["num_heads"]
    head_dim = p["head_dim"]
    rope_head_dim = p["rope_head_dim"]
    max_position = p["max_position"]

    torch.manual_seed(37)
    o = torch.randn(
        (num_tokens, num_heads, head_dim), device="cuda", dtype=torch.bfloat16
    )
    positions = torch.randint(
        0, max_position, (num_tokens,), device="cuda", dtype=torch.int64
    )
    # Layout required by the kernel: [P, rope_head_dim] holding cos | sin,
    # each half of width rope_head_dim // 2.
    half = rope_head_dim // 2
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, half, device="cuda", dtype=torch.float32) / half)
    )
    pos = torch.arange(max_position, device="cuda", dtype=torch.float32)
    angles = pos[:, None] * inv_freq[None, :]
    cos_sin_cache = torch.cat([angles.cos(), angles.sin()], dim=1).to(torch.bfloat16)

    return {
        "cfg": case,
        "module": _load_kernel_module(),
        "o": o,
        "positions": positions,
        "cos_sin_cache": cos_sin_cache,
        "rope_head_dim": rope_head_dim,
    }


def _run(inputs: dict):
    return inputs["module"]._fused_inverse_rope_gptj(
        inputs["o"],
        inputs["positions"],
        inputs["cos_sin_cache"],
        inputs["rope_head_dim"],
    )


def _reference(inputs: dict):
    """Inverse GPT-J RoPE in float32.

    The leading NoPE region passes through unchanged. On the trailing rope
    region the interleaved (even, odd) pairs are counter-rotated:
    ``out_even = a*cos + b*sin``, ``out_odd = b*cos - a*sin``.
    """
    torch = _torch()
    o = inputs["o"].float()
    rope_dim = inputs["rope_head_dim"]
    nope = o.shape[-1] - rope_dim
    half = rope_dim // 2

    cs = inputs["cos_sin_cache"].float()[inputs["positions"]]  # (T, rope_dim)
    cos = cs[:, :half][:, None, :]  # (T, 1, half)
    sin = cs[:, half:][:, None, :]

    out = o.clone()
    rope = o[..., nope:]
    a = rope[..., 0::2]
    b = rope[..., 1::2]
    out[..., nope:][..., 0::2] = a * cos + b * sin
    out[..., nope:][..., 1::2] = b * cos - a * sin
    return out.to(torch.bfloat16)


def _assert_close(inputs: dict, got) -> None:
    _torch().testing.assert_close(got, _reference(inputs), atol=2e-2, rtol=2e-2)


def _perturb_inputs(inputs: dict) -> None:
    """Refresh data inputs in place with values no earlier launch has seen.

    ``positions`` and the cos/sin cache are structure, not data, so they stay
    fixed; only the activation is redrawn.
    """
    torch = _torch()
    torch.manual_seed(59)
    inputs["o"].normal_()


def _compile_smoke_case(case: dict) -> dict:
    """Shrink a case so the compile smoke test stays cheap. compile only."""
    smoke = {**case, "params": dict(case["params"])}
    smoke["params"]["num_tokens"] = min(case["params"]["num_tokens"], 8)
    smoke["params"]["num_heads"] = min(case["params"]["num_heads"], 4)
    smoke["params"]["max_position"] = min(case["params"]["max_position"], 512)
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
            "regime": case.get("regime"),
            "observed_workitem_grid": case.get("observed_workitem_grid"),
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
