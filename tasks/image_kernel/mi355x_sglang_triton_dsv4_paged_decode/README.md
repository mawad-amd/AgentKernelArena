# mi355x_sglang_triton_dsv4_paged_decode

DeepSeek-V4-Flash decode attention as SGLang ships it on MI355X. AMD's nightly CI pins
`SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton`, which routes decode through
`kernels/ops/attention/dsv4/unified_kv_kernels/paged_decode.py` — a split-K kernel writing
`(m, l, acc)` partials plus a reduce kernel that combines them and folds the attention sink.

## Cases

The three attention-**compression** layer classes. Counts from the checkpoint's
`compress_ratios` (44 entries: 43 layers + MTP) and confirmed against a live trace.

| case | class | layers | attended KV |
|---|---|---|---|
| `c4-decode` | ratio 4 | 21 | 128 window + `index_topk` 512 = 640 |
| `c128-decode` | ratio 128 | 20 | 128 window + ~seq/128 = 281 |
| `swa-decode` | ratio 0 | 2 (indices 0, 1) | 128 window only |

**Do not relabel these by routing class.** Routing is a different partition
(`num_hash_layers=3` → layers 0, 1, 2). Layer 2 is hash-routed *and* C4.

## Measured geometry (MI355X, rocprofv3, eager)

`Grid_Size` in rocprofv3 is **work-items, not workgroups** — divide by `Workgroup_Size`
before reasoning about occupancy.

| kernel | work-items | wg | workgroups | VGPR | SGPR |
|---|---|---|---|---|---|
| split | (512,1,64) | 512 | **64** | 124 / 108 | 64 |
| reduce | (256,64,8) | 256 | 512 | 16 | 48 |

Two VGPR classes on split, 22 dispatches/token at 124 (20 C128 + 2 ratio-0) and 21 at 108
(the C4 layers). Grid, VGPR and the 22:21 ratio are byte-identical at 128, 1024 and 16384
prefill tokens — geometry varies by layer class only, never by context.

Aggregation key for any geometry comparison:
`grid(x,y,z) + workgroup(x,y,z) + VGPR + Accum_VGPR + SGPR + LDS + Scratch`. Keying on grid
alone shows one class here and hides the register split.

## Physics you are up against

At batch 1 the base grid is **one workgroup**; split-K takes it to 64 on a 256-CU part. So
split-K is mandatory and not sufficient — three quarters of the device is idle.

The obvious move is more splits. It has been measured and it loses. `kv_splits` is capped at
`_MAX_KV_SPLITS = 64` while the kernel's own heuristic computes a device-fill target of
2 WG/CU = 512. Raising it 64 → 256 (ABBA, rocprofv3, GPU time of split+reduce, treatment
verified via `Grid_Size_Z`):

```
             split_us   reduce_us    sum_us
ks=64  mean   13606.0      4764.7   18370.7
ks=256 mean   14027.9      6206.1   20234.0     +10.1%
```

+3.1% on split, **+30.2% on reduce**. The cap is priced, not overlooked.

The heuristic is also **CUDAGraph-safe by design** and must stay that way: it may not read
any tensor value or shape, because in production `kv_indices.shape[0]` is a padded bucket
unrelated to true per-token kv_len, and a heuristic reading it would mis-tune at capture
time. Any change that makes split selection data-dependent breaks graph replay.

## Provenance

Traced on a single MI355X (gfx950), one GPU pinned for the run, image
`rocm/sgl-dev:v0.5.18-rocm720-mi35x-20260825`, TP=1, batch 1, eager. Checkpoint verified
byte-exact against the HuggingFace blob manifest (47/47 files).

Decode kernel identity comes from an **eager** run: graph-replayed decode is unobservable —
rocprofv3 aborts on replay with `HSA_STATUS_ERROR_INVALID_PACKET_FORMAT` and Kineto returns
one kernel event. Capture-only launch structure (multi-stream overlap) is therefore
**unobserved, not absent**.

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
| kernels under optimization — `_paged_decode_split_kernel`, `_paged_decode_reduce_kernel`, `_sparse_attn_v4_paged_decode_triton` | **SGLang**, `python/sglang/kernels/ops/attention/dsv4/unified_kv_kernels/paged_decode.py` (sgl-project/sglang) | Apache-2.0 |
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
