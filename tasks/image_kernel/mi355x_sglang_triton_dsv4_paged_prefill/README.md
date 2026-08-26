# mi355x_sglang_triton_dsv4_paged_prefill

DeepSeek-V4-Flash prefill attention as SGLang ships it on MI355X. AMD's nightly CI pins
`SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton`, which routes prefill through
`kernels/ops/attention/dsv4/unified_kv_kernels/paged_prefill.py`.

The kernel attends over **two KV sources in one pass**: the paged prefix already resident in
the unified pool, and this forward's flat extend rows that are not in the ring yet. Both
index streams carry `-1` sentinels which are skipped. It is a single attention over the
union of the two sources, not two attentions merged.

## Cases

The three attention-**compression** layer classes, differing only in prefix length.

| case | class | layers | prefix | extend |
|---|---|---|---|---|
| `c4-prefill` | ratio 4 | 21 | 512 (`index_topk`) | 128 |
| `c128-prefill` | ratio 128 | 20 | 153 (~seq/128) | 128 |
| `swa-prefill` | ratio 0 | 2 (indices 0, 1) | 0 | 128 |

**Do not relabel by routing class.** Routing is a different partition of the same 43 layers
(`num_hash_layers=3` → layers 0, 1, 2); layer 2 is hash-routed *and* C4.

## Provenance

Traced on a single MI355X (gfx950), one GPU pinned for the run, image
`rocm/sgl-dev:v0.5.18-rocm720-mi35x-20260825`, TP=1, batch 1, eager. Checkpoint verified
byte-exact against the HuggingFace blob manifest (47/47 files).

The unified-KV prefill index and store kernels — `_build_prefill_indices_kernel`,
`_prefill_lengths_kernel`, `_swa_scatter_kernel`, `_init_compressed_attn_metadata_kernel` —
each fire **43 times, once per layer**, which is the per-layer confirmation for this path.

## What is actually running, and the occupancy wall

On this image the startup log says
`Disabling SGLANG_OPT_FLASHMLA_SPARSE_PREFILL by default on ROCm/HIP for DeepseekV4ForCausalLM`.
So in the shipped configuration the Triton path here owns the **index construction and
store**, while the attention math goes to an asm kernel, `pa_prefill_16mx8_32nx1_kernel`.
That kernel is occupancy-walled and worth knowing before optimising either side:

```
work-items (65536,1,1)  wg=512  ->  128 workgroups on 256 CUs
LDS 135168 B/WG  (132 KB of gfx950's 160 KB)   vgpr=128   scratch=108
```

Half the device is idle and the occupied half runs one workgroup per CU with register
spill. That LDS figure was **predicted at ~136 KB from DeepSeek's own TileLang `sparse_attn`
reference before it was measured**, so the pressure is intrinsic to the operator at
h=64 / d=512 — not an artefact of a NVIDIA-tuned reference.

`Grid_Size` in rocprofv3 is **work-items, not workgroups**. Divide by `Workgroup_Size`
before reasoning about occupancy. Aggregation key for any geometry comparison:
`grid(x,y,z) + workgroup(x,y,z) + VGPR + Accum_VGPR + SGPR + LDS + Scratch` — the
discriminating column is per-kernel and cannot be guessed in advance.

## Known coverage limit: the attention sink is exercised but not verified

`attn_sink` is built at production shape — the checkpoint ships 44 sink tensors, 64 values
per layer, roughly N(0.5, 0.4) — and the reference folds `exp(sink - m)` into the softmax
denominator. The kernel demonstrably applies it (sweep on `c4-decode`, kv_len 640):

| sink | max abs delta vs sink=0 | mean abs output |
|---|---|---|
| 0.0 | 0.000000 | 0.002994 |
| 5.0 | 0.001953 | 0.002434 |
| 50.0 | 0.010254 | **0.000000** |

At 50 the sink consumes the whole denominator and the output collapses to exactly zero, so
the path is live and correctly wired.

**But it is not verified at production magnitude.** Dropping the sink from the reference
only, leaving it in the kernel, still passes: relerr moves 0.0021 → 0.0031 / 0.0048 /
0.0094 against a 0.04 tolerance. An agent could break the sink subtly and this task would
not notice.

This is arithmetic, not a fixable oversight. With 512+ attended keys an O(1) sink is
~0.2-0.3% of the denominator, below a tolerance that is set by bf16 error rather than
chosen. Note also that **widening the score spread makes it relatively worse, not better** —
max-logit grows with the spread, so a fixed-magnitude sink becomes a smaller share of the
denominator. Tried and reverted. The binding constraint is the key count.

Verifying it would need either a tolerance tighter than bf16 permits, or a synthetic
high-sink case that no longer reflects the trace. Faithful-and-labelled was chosen over
verified-and-fake.

## Source of the code under optimization

**No SGLang or vLLM source is vendored into this task.** The kernels live in the container
image and are seeded into the workspace at run time from `image_repo_path`; this directory
contains only the harness.

| what | where it comes from | licence |
|---|---|---|
| kernels under optimization — `_sparse_attn_v4_paged_prefill_kernel`, `_sparse_attn_v4_paged_prefill_triton` | **SGLang**, `python/sglang/kernels/ops/attention/dsv4/unified_kv_kernels/paged_prefill.py` (sgl-project/sglang) | Apache-2.0 |
| harness (`scripts/`), config, cases, this README | written for this task, from the AgentKernelArena template | Apache-2.0 (repo) |
| correctness reference maths | contracts read from DeepSeek's `inference/{kernel,model}.py`, shipped in [deepseek-ai/DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash); re-implemented in PyTorch, not copied from their TileLang | MIT |

## Known issue: `forge --bench-mode` fails in the launcher's invocation shape

Not a defect in this task. `scripts/task_runner.py` imports the injected benchmark helper as
`from _aka_benchmark import ...`, and `_aka_benchmark.py` is materialized into `scripts/`.
That resolves when you run `python3 scripts/task_runner.py`, because Python puts the
script's directory on `sys.path`. **Arena's forge launcher copies `forge_driver.py` to the
workspace root and runs it there**, so `scripts/` is not on the path and the import fails:

```
forge --bench-mode  from scripts/        RC=0
forge --bench-mode  at workspace root    RC=1   ModuleNotFoundError: No module named 'src.tools'
```

`compile`, `correctness`, `performance`, `--mode full` and `--profile-run` are unaffected.

Introduced by [PR #77](https://github.com/AMD-AGI/AgentKernelArena/pull/77) (merged to main
2026-08-25, `aae00b78`), which moved the injected helper from code inlined in each
`task_runner.py` to a separate imported module. **16 tasks on `main` are affected, all of
which predate this one.**

Confirmed by before/after on this task across the boundary, code unchanged:

| revision | materialization | bench-mode at workspace root |
|---|---|---|
| `c801c405` (pre-#77) | helper inlined, no `_aka_benchmark.py` | RC=0 |
| `aae00b78` (post-#77) | shim + `_aka_benchmark.py` | **RC=1** |

**No local workaround is applied here, deliberately.** A per-task `sys.path` insert would
make this task pass whether or not the real fix lands, so it could not verify the fix — and
it would exclude this task from any grep for affected tasks. The fix belongs in
`src/tools/perf/vllm_cuda_graph_block.py`, which is injected everywhere.
