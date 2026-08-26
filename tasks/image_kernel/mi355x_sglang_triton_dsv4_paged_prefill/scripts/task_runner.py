#!/usr/bin/env python3
"""Image-kernel harness for SGLang DeepSeek-V4 unified-KV paged decode attention.

Target device kernel : ``_sparse_attn_v4_paged_prefill_kernel``
Timed launcher       : ``sparse_attn_v4_paged_prefill``
Source               : sglang/kernels/ops/attention/dsv4/unified_kv_kernels/paged_prefill.py

This is the prefill attention SGLang ships on MI355X under
``SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton``. It attends over **two** KV sources in
one pass: the paged prefix already in the unified pool, and this forward's flat extend
rows that are not in the ring yet. Both index streams carry -1 sentinels that are skipped.

Cases are the three DeepSeek-V4-Flash attention-compression layer classes, which
determine the attended KV length per layer. Counts confirmed against the checkpoint's
``compress_ratios`` (44 entries, 43 real layers + MTP) and against dispatch counts in a
live MI355X trace:

  c4    21 layers  ratio 4    sliding window 128 + index_topk 512  -> 640 attended
  c128  20 layers  ratio 128  sliding window 128 + ~seq/128        -> 281 attended
  swa    2 layers  ratio 0    sliding window 128 only              -> 128 attended

Note the layer classes here are the *compression* partition. It is not the same cut as
the routing partition (``num_hash_layers=3`` -> layers 0,1,2); layer 2 is hash-routed and
C4. Do not relabel these cases by routing class.

MQA: ``num_key_value_heads=1``, so ``unified_kv`` serves as both K and V.
``attn_sink`` is a learnable per-head logit whose value vector is zero -- it removes mass
from the softmax denominator only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
SPEC = json.loads((WORKSPACE / "session_cases.json").read_text())
OPERATOR = SPEC["operator"]
CASES = SPEC["cases"]


def _configure() -> None:
    for key in ("GPU_ARCHS", "PYTORCH_ROCM_ARCH", "AMDGPU_TARGETS", "GPU_TARGETS"):
        os.environ.setdefault(key, "gfx950")
    # The agent edits the workspace-seeded copy, so that copy MUST be what imports.
    # This used to fall back to the in-image install when the seed was missing, which
    # made a seeding failure look like a clean run against unmodified code: the agent's
    # edits vanish and the result reads as "no optimisation found" rather than "broken
    # harness". Verified by poisoning the in-image kernel -- with the seed absent the
    # installed copy ran and correctness still passed. Fail closed instead.
    seeded = WORKSPACE / "sglang"
    if (seeded / "__init__.py").is_file():
        sys.path.insert(0, str(WORKSPACE))
    elif os.environ.get("AKA_ALLOW_IN_IMAGE_SGLANG") == "1":
        # Explicit opt-in for standalone/dev runs outside Arena. Never silent.
        print(
            "WARNING: no seeded sglang/ in the workspace; running the IN-IMAGE install. "
            "Edits to the seeded copy will NOT take effect.",
            file=sys.stderr,
        )
        sys.path.insert(0, os.environ.get("SGLANG_PYTHON", "/sgl-workspace/sglang/python"))
    else:
        raise RuntimeError(
            f"no seeded sglang/ package at {seeded} -- refusing to run against the "
            "in-image install, because that would score unmodified code as a clean pass. "
            "Arena seeds image_repo_path into the workspace; for a deliberate standalone "
            "run set AKA_ALLOW_IN_IMAGE_SGLANG=1."
        )
    os.chdir(WORKSPACE)


def _torch():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("ROCm GPU (gfx950) is required")
    return torch


def _relerr(a, b) -> float:
    a = a.float()
    b = b.float()
    return float(((a - b).norm() / (b.norm() + 1e-8)).item())


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


# --------------------------------------------------------------------------- #
# Inputs / call / reference
# --------------------------------------------------------------------------- #
def _make(case: dict, correctness: bool = False) -> dict:
    """Build one case. ``correctness`` is part of the forge_driver contract
    (``--profile-run`` passes it); this operator builds identical inputs either way,
    so it is accepted and unused rather than silently rejected."""
    torch = _torch()
    p = case["params"]
    T, H, D = p["t"], p["h"], p["d"]
    prefix_len, extend_len = p["prefix_len"], p["extend_len"]
    torch.manual_seed(case.get("seed", 0))

    # Unit variance is derived, not guessed. DeepSeek's own reference applies a
    # WEIGHTLESS per-head RMS norm to q immediately before attention
    #   (inference/model.py:498  q *= rsqrt(q.square().mean(-1) + eps))
    # so every q head-vector has RMS exactly 1, and kv goes through kv_norm
    # (RMSNorm with an O(1) learned weight). Score std is sigma_q*sigma_kv*sqrt(D)*scale
    # and sqrt(D)*scale == 1, so trained attention here sits at score std ~1.
    # The template's 0.1 gives 0.01 -- 512 near-identical logits and a degenerate
    # softmax that a kernel could exploit (e.g. skipping the max subtraction).
    n_pages = max(T * max(prefix_len, 1) * 2, 4096)
    unified_kv = torch.randn(n_pages, D, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(max(T * extend_len, 1), D, device="cuda", dtype=torch.bfloat16)
    q = torch.randn(T, H, D, device="cuda", dtype=torch.bfloat16)

    def _csr(n_per_row: int, values):
        # torch.arange rejects step=0, and the ratio-0 class legitimately has an empty
        # prefix. Build the indptr from a multiply so len==0 stays a valid empty CSR
        # rather than an exception.
        ptr = (torch.arange(T + 1, device="cuda", dtype=torch.int32) * n_per_row)
        return values.contiguous(), ptr

    pidx, pptr = _csr(
        prefix_len,
        torch.randperm(n_pages, device="cuda")[: T * prefix_len].to(torch.int32),
    )
    eidx, eptr = _csr(
        extend_len, torch.arange(T * extend_len, device="cuda", dtype=torch.int32)
    )
    attn_sink = torch.randn(H, device="cuda", dtype=torch.float32) * 0.4 + 0.5
    return {
        "cfg": case,
        "q": q,
        "unified_kv": unified_kv,
        "kv_indices_prefix": pidx,
        "kv_indptr_prefix": pptr,
        "kv": kv,
        "kv_indices_extend": eidx,
        "kv_indptr_extend": eptr,
        "attn_sink": attn_sink,
        "softmax_scale": float(D) ** -0.5,
    }


def _run(inputs: dict):
    from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.paged_prefill import (
        sparse_attn_v4_paged_prefill,
    )

    return sparse_attn_v4_paged_prefill(
        inputs["q"],
        inputs["unified_kv"],
        inputs["kv_indices_prefix"],
        inputs["kv_indptr_prefix"],
        inputs["kv"],
        inputs["kv_indices_extend"],
        inputs["kv_indptr_extend"],
        inputs["attn_sink"],
        inputs["softmax_scale"],
    )


def _reference(inputs: dict):
    """Dense fp32 attention over prefix pages + extend rows, with the zero-valued sink.

    Both index streams may carry -1 sentinels, which the kernel skips; the reference
    drops them the same way. The two sources are concatenated before the softmax -- it is
    one attention over the union, not two attentions merged.
    """
    torch = _torch()
    q = inputs["q"].float()
    pool = inputs["unified_kv"].float()
    ext = inputs["kv"].float()
    pi, pp = inputs["kv_indices_prefix"], inputs["kv_indptr_prefix"].tolist()
    ei, ep = inputs["kv_indices_extend"], inputs["kv_indptr_extend"].tolist()
    sink = inputs["attn_sink"].float()
    scale = inputs["softmax_scale"]

    T, H, D = q.shape
    out = torch.empty(T, H, D, device=q.device, dtype=torch.float32)
    for t in range(T):
        a = pi[pp[t] : pp[t + 1]]
        b = ei[ep[t] : ep[t + 1]]
        a = a[a >= 0].long()
        b = b[b >= 0].long()
        sel = torch.cat([pool.index_select(0, a), ext.index_select(0, b)], dim=0)
        logits = (q[t] @ sel.T) * scale
        m = torch.maximum(logits.max(dim=-1).values, sink)
        pr = torch.exp(logits - m[:, None])
        denom = pr.sum(dim=-1) + torch.exp(sink - m)
        out[t] = (pr @ sel) / denom[:, None]
    return out.to(inputs["q"].dtype)


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
def _assert_close(case: dict, inputs: dict, got, label: str = "") -> float:
    err = _relerr(got, _reference(inputs))
    tol = case["params"].get("max_relerr", 0.06)
    assert err < tol, (case["id"], label, err, tol)
    return err


def _perturb_inputs(inputs: dict) -> None:
    """Refresh q and both KV sources in place. Index streams are left alone: the ragged
    structure is what the case is defined by."""
    torch = _torch()
    torch.manual_seed(59)
    inputs["q"].copy_(torch.randn_like(inputs["q"]))
    inputs["unified_kv"].copy_(torch.randn_like(inputs["unified_kv"]))
    inputs["kv"].copy_(torch.randn_like(inputs["kv"]))


def _assert_timed_outputs(case: dict, inputs: dict, timed) -> None:
    """Validate the invocation the benchmark actually timed.

    ``run_correctness`` checks a separate call, which a kernel can tell apart
    from the scored one. This re-runs the timed unit against a freshly perturbed
    activation and checks the buffer it wrote, so work that the scored path skips
    cannot hide behind a correctness call that took a different branch.
    """
    if not timed.bound:
        raise RuntimeError("benchmark did not expose the timed invocation")
    _perturb_inputs(inputs)
    if timed.outputs is not None:
        timed.outputs.fill_(float("nan"))
    _assert_close(case, inputs, timed.rerun(), label="timed run")


def _compile_smoke_case() -> dict:
    """Shrunk shape for the compile gate only. Correctness and performance always run
    the full scored shape -- Triton can pick different tiles at different sizes, so a
    small shape passing proves nothing about the scored one."""
    case = json.loads(json.dumps(CASES[0]))
    case["id"] = case["id"] + "-smoke"
    case["params"]["prefix_len"] = 32
    case["params"]["extend_len"] = 8
    return case


def run_compile() -> None:
    inputs = _make(_compile_smoke_case())
    _run(inputs)
    _torch().cuda.synchronize()
    print("dsv4_paged_prefill compile smoke: PASS")


def run_correctness() -> None:
    torch = _torch()
    for case in CASES:
        inputs = _make(case)
        got = _run(inputs)
        torch.cuda.synchronize()
        err = _assert_close(case, inputs, got)
        print("correctness PASS", case["id"], f"relerr={err:.4f}")


def run_performance() -> None:
    torch = _torch()
    rows = []
    for case in CASES:
        inputs = _make(case)
        _run(inputs)
        torch.cuda.synchronize()
        timed = _TimedRun()
        ms, bmeta = _benchmark_cuda_graph_or_events(lambda: _run(inputs), timed_run=timed)
        _assert_timed_outputs(case, inputs, timed)
        row = {
            "test_case_id": case["id"],
            "execution_time_ms": ms,
            "metadata": {**case["params"], "family": case.get("family"),
                         "regime": case.get("regime"), **bmeta},
        }
        rows.append(row)
        print(case["id"], f"{ms:.6f} ms", bmeta.get("benchmark_method"),
              bmeta.get("benchmark_fallback_reason", ""))
    out = WORKSPACE / "build"
    out.mkdir(parents=True, exist_ok=True)
    (out / "performance_report.json").write_text(json.dumps(rows, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["compile", "correctness", "performance", "manifest"])
    mode = parser.parse_args().mode
    if mode == "manifest":
        print(json.dumps(SPEC, indent=2))
        return
    _configure()
    {"compile": run_compile, "correctness": run_correctness, "performance": run_performance}[mode]()


if __name__ == "__main__":
    main()
