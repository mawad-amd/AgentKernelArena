# mi355x-dsv4flash-inverse-rope-gptj-20260826

Fused inverse GPT-J RoPE on the DeepSeek-V4-Flash attention output-projection path.
Scores the Triton kernel `_inverse_rope_gptj_kernel` through its launcher
`_fused_inverse_rope_gptj`.

## Why this kernel

One of the highest launch-count kernels in the model: **4,902 launches in a measured prefill** and
**11,008 in a 256-token decode** (43 layers x 256 tokens). It carries 3.64% of prefill GPU time in
the whole-process trace despite being arithmetically trivial — it copies the NoPE region unchanged
and counter-rotates only the trailing 64 rope elements.

Nothing in Arena covered it.

## Provenance

rocprofv3 `--kernel-trace --marker-trace` on `vllm/vllm-openai-rocm:v0.27.0`, TP=1, batch 1, eager,
windowed by a roctx range around the measured generate with warmup outside. Venue
MI355X gfx950, ROCm 7.2.2, single dedicated GPU.

The launch is one program per `(token, head)` at 256 threads. Two distinct geometries appear in each
phase, so the two cases cover both regimes rather than one:

| case | regime | observed work-item grid | tokens | heads |
|---|---|---|---|---|
| `dsv4-flash-prefill-t543` | bandwidth-bound | (139008,64,1) | 543 | 64 |
| `dsv4-flash-decode-t1` | launch-bound | (256,64,1) | 1 | 64 |

**rocprofv3 `Grid_Size` is work-items, not workgroups.** 139008 / 256 = 543 programs. Reading it as
workgroups overstates the launch by 256x.

At decode the whole kernel is 64 programs on 256 CUs, so it is pure launch overhead — 0.0019 ms for
a rotation on 64x64 bf16 elements. That is the interesting half of this task: the prefill case
rewards bandwidth work, the decode case rewards not launching at all, and the two pull in different
directions.

## Correctness

Reference is float32 and written from the rotation definition, not from the kernel: NoPE passes
through, and on the rope region the interleaved `(even, odd)` pairs become
`out_even = a*cos + b*sin`, `out_odd = b*cos - a*sin`, with cos/sin gathered from a
`[P, rope_head_dim]` cache indexed by token position (`cos | sin`, each half `rope_head_dim // 2`
wide). Tolerance atol/rtol 2e-2 for bf16.

`positions` and the cos/sin cache are structure rather than data, so `_perturb_inputs` redraws only
the activation — perturbing the positions would change which cache rows are read and invalidate the
comparison.

## Measured baseline

`vllm/vllm-openai-rocm:v0.27.0`, MI355X gfx950, single dedicated GPU:

```
compile      inverse_rope_gptj compile smoke: PASS
correctness  PASS dsv4-flash-prefill-t543
             PASS dsv4-flash-decode-t1
performance  dsv4-flash-prefill-t543   0.025211 ms  cuda_graph
             dsv4-flash-decode-t1      0.001903 ms  cuda_graph
```

Both timed under `cuda_graph`, not the event fallback, and `_assert_timed_outputs` passed — the
timed invocation was re-run against perturbed inputs with NaN-filled output buffers and still
matched.

## Caveats

`evaluator.py` means over cases, and these two cases are in opposite regimes with a 13x latency
difference. A change that wins on prefill can lose on decode and still improve the mean. Read the
per-case times in `performance_report.json`.

Launch counts and geometry are from eager execution; graph-replayed decode is unobservable by
rocprofv3 (aborts or degrades) and by Kineto (cannot see into a replay).

## Sources

- **DeepSeek-V4-Flash reference** — `inference/model.py` in the weights repo, read for the
  normalisation contract. The correctness reference here is written independently in torch from the
  rotation definition; only the contract came from that source.
