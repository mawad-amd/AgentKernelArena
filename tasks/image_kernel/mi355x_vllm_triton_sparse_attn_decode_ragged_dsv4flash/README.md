# mi355x-dsv4flash-sparse-attn-decode-ragged-20260826

DeepSeek-V4-Flash sparse MLA **decode** on vLLM/ROCm. Scores the Triton split-KV pair
`_sparse_attn_decode_partial_kernel` + `_sparse_attn_decode_reduce_kernel` through their launcher
`_rocm_sparse_attn_decode_ragged_triton`. Both stages are scored together, so moving work between
split and reduce is a legal optimisation provided the pair stays correct.

## Provenance

Cases are the **three attention-compression layer classes** measured on a real V4-Flash decode, not
synthetic sizes. rocprofv3 `--kernel-trace --marker-trace` on `vllm/vllm-openai-rocm:v0.27.0`,
TP=1, batch 1, eager, windowed by a roctx range around the measured generate with warmup outside.
MI355X gfx950, ROCm 7.2.2, single dedicated GPU.

Distinguished under the full aggregation key — grid(x,y,z) + workgroup(x,y,z) + VGPR + Accum_VGPR +
SGPR + LDS + Scratch. A coarser key merges them: the c4/c128 pair differs only in grid and the
residual class only in registers, so no smaller key separates all three.

| case | layer class | layers | dispatches/token | num_splits | VGPR |
|---|---|---|---|---|---|
| `dsv4-flash-decode-c4-topk512` | indexed, `compress_ratio 4` | 21 of 43 | 21 | 16 | 176 |
| `dsv4-flash-decode-c128-compressed` | compressed, `compress_ratio 128` | 20 of 43 | 20 | 8 | 176 |
| `dsv4-flash-decode-swa-window128` | residual, `compress_ratio 0` | layers 0-1 | 2 | 4 | 168 |

**Validity check:** dispatches per class divide to exact integers per token — 21.00 / 20.00 / 2.00,
summing to the 43 layers. A non-integer means the trace window is wrong or classes are mixing.

The taxonomy is confirmed three further ways from launch counts alone: the indexer GEMM fires 21x,
`_inverse_rope_gptj_kernel` splits 43-per-layer vs per-layer-per-token, and `compress_ratios` in
`config.json` reads 21/20/2 over the 43 real layers.

## Physics an optimiser should know

**K is bounded by the architecture, at any context length.** `index_topk = 512` caps the c4 layers
at 512 attended tokens regardless of sequence length; the c128 layers see `ceil(ctx/128)` of a
compressed cache; the residual layers use `sliding_window = 128`. A 19,601-token prompt produced
launch geometry identical to a 169-token prompt. **Long context does not produce long K in this
model** — the sparse indexer is the mechanism that prevents it.

That kills the obvious idea. `_decode_num_splits` caps its search at 16 splits; widening it to 64
changes the chosen count only for `avg_main_len >= ~4096`, which this model never reaches. Measured
directly against the deployed heuristic:

| avg_main_len | cap 16 | cap 64 |
|---|---|---|
| 512 | 16 | 16 |
| 4096 | 16 | 64 |
| 19600 | 16 | 62 |

So raising the split cap is inert here. The cap is tuned for short K and short K is the only regime
this architecture has.

**Occupancy is the standing headroom.** At batch 1 on 256 CUs the three classes launch 64 / 32 / 16
workgroups — 25% / 12.5% / 6.25% of the device. Split-KV fixes the catastrophic one-workgroup case
but does not fill the GPU. The heuristic's own target is ~1 workgroup per CU and it ships at a
quarter of that. Note the counts below the cap come from a snap-down that only fires when a smaller
split yields the same wave count *and* the same `BLOCK_K` iteration count, so those are not wasted
parallelism — extra splits there add reduce and HBM traffic for no iteration saving.

## Correctness

Reference is an independent float32 attention: dequantise the selected `fp8_ds_mla` rows, then
`softmax(q·Kᵀ · scale) @ K` — the latent is both K and V. It is written from the cache layout and
the model contract, not derived from the kernel, so agreement means the two agree on the maths
rather than by construction.

The cache is built with vLLM's own `quantize_and_insert_k_cache`, so the bytes the kernel reads are
what production writes: 576 bytes per token (448 NoPE fp8 + 64 RoPE bf16) plus 8 bytes carrying 7
e8m0 group scales, one per 64 NoPE elements.

**Both q and kv are unit variance, and that is derived from the model rather than chosen.**
DeepSeek applies a *weightless* per-head RMS norm immediately before attention —
`q *= rsqrt(q.square().mean(-1) + eps)` in the reference `inference/model.py`, and vLLM documents
the same at `deepseek_v4/attention.py:634` ("per-head RMSNorm (no weight)"). So every q head-vector
has RMS exactly 1: **sigma_q = 1**. `kv` passes through `kv_norm`, an RMSNorm with an O(1) learned
weight, so sigma_kv is order 1; its magnitude is safe for fp8 because the cache carries a per-64
e8m0 scale that absorbs the range. Score std is `sigma_q * sigma_kv * sqrt(D) * scale` and
`sqrt(D)*scale == 1`, so trained attention here sits at **score std ~1**.

Earlier revisions of this task used 0.125 (score std 0.016 — 512 near-identical logits and a flat
softmax a kernel could exploit by skipping the max subtraction) and then 8.0 (overshoot, justified
by a sink argument that was wrong — see below). The architecture pins it at 1.

## Attention sink — exercised, but not independently verified

DeepSeek-V4 attention has a learnable per-head sink: an extra logit whose value vector is zero, so
it removes mass from the softmax denominator without contributing to the output. The V4-Flash
checkpoint carries **44 `attn_sink` tensors** (43 layers + MTP) and vLLM passes
`attn_sink=self.attn_sink` on every decode call (`deepseek_v4/amd/rocm.py:702,831`) — it is never
`None` in production. The harness therefore builds a real sink at checkpoint magnitudes: 64 values,
sampled to mean 0.5 / std 0.5, matching layers 0-3 of the checkpoint (measured mean 0.35-0.62,
std 0.20-0.64).

**Score spread does not make the sink detectable, and an earlier version of this file claimed it
did.** Widening the spread makes things marginally *worse*, because the sink's share of the denominator is
`exp(sink - m) / (sum_k exp(s_k - m) + exp(sink - m))` and `m` is the max logit, which grows with
sigma. At flat scores the share is about 0.3%; at unit std it is about 0.19%. The binding constraint
is the **key count**, not the spread: with 512+ attended keys an O(1) sink is a fraction of a percent
of the denominator at any sigma. Credit to the SGLang-side review for catching this — the same
widening was tried there and reverted.

**The kernel does apply it.** Sweeping the sink with everything else fixed:

| sink | max abs delta vs baseline | mean abs output |
|---|---|---|
| 0.0 | 0.000244 | 0.008189 |
| 5.0 | 0.006592 | 0.006951 |
| 50.0 | 0.041748 | **0.000000** |

At 50 the sink absorbs the entire softmax denominator and the output collapses to zero, which is
exactly the defined behaviour.

**But at production magnitudes the effect is below the tolerance this task can use.** Dropping the
sink from the reference while leaving it in the kernel still passes: with 512 attended keys the
denominator is large enough that a sink near 0.5 shifts the output by well under the 2e-2 atol/rtol,
and that tolerance is set by fp8 quantisation error, not chosen. So the sink branch is **executed
and shaped like production, but an agent could subtly break it without this task noticing**. A gross
break would need a sink magnitude the model does not use.

Recorded rather than papered over: adding the sink made the task production-faithful, it did not
make the sink verified. Verifying it would need either a tighter tolerance than fp8 permits or a
synthetic high-sink case that no longer reflects the trace.

## Measured baseline

`vllm/vllm-openai-rocm:v0.27.0`, MI355X gfx950, single dedicated GPU:

```
compile      sparse_attn_decode_ragged compile smoke: PASS
correctness  PASS dsv4-flash-decode-c4-topk512
             PASS dsv4-flash-decode-c128-compressed
             PASS dsv4-flash-decode-swa-window128
performance  dsv4-flash-decode-c4-topk512        0.013707 ms  cuda_graph
             dsv4-flash-decode-c128-compressed   0.012149 ms  cuda_graph
             dsv4-flash-decode-swa-window128     0.012121 ms  cuda_graph
forge        --mode full RC=0 allclose: True
             --bench-mode RC=0 mean_ms: 0.012664
             --profile-run RC=0
```

All three timed under `cuda_graph`, not the event fallback, and `_assert_timed_outputs` passed —
the timed invocation was re-run against perturbed inputs with NaN-filled output buffers and still
matched the reference.

## Caveats

**Per-case regression can hide.** `evaluator.py` means over cases, and VGPR differs across classes
(176/176/168), so occupancy is a per-case property and a win on one class can mask a regression on
another. Read the per-case times in `performance_report.json`, not just the mean.

**`decode_gpu_share_eager` is not Arena `gpu_pct`.** It is a share of summed GPU time over
decode-phase kernels under **eager** execution, with the denominator recorded beside it. It is not
comparable to Hyperloom production percentages. Graph-replayed decode is unobservable — rocprofv3
aborts or silently degrades to eager, and Kineto cannot see into a replay — so eager is the only
observable regime and the number is labelled accordingly.

**Launch geometry is eager-specific.** `adaptive_splits` is gated on `for_cudagraph_capture`
(`deepseek_v4/amd/rocm.py:639`), so a different split selector runs under capture. The layer-class
taxonomy is architectural and holds; the class-to-grid mapping recorded above is measured eager and
is not verified under capture.

**rocprofv3 `Grid_Size` is work-items, not workgroups.** Divide by the workgroup size. Reading it
as workgroups understates occupancy by 256x and hides the finding above.

## Sources

- **vLLM** (Apache-2.0) — the fp8_ds_mla cache pack/unpack logic in `scripts/task_runner.py` follows
  `tests/kernels/attention/test_rocm_triton_attn_dsv4.py`, so the bytes fed to the kernel match what
  vLLM's production store path writes.
- **DeepSeek-V4-Flash reference** — `inference/model.py` in the weights repo, read for the
  input-scaling contract (weightless per-head RMS norm on q, hence sigma_q = 1). The correctness
  reference here is written independently in torch; only the contract came from that source.
