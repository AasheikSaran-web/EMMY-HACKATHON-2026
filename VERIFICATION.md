# Hidden-set verification (post-event)

The hackathon's hosted runner (`runner.riftstack.ai:48701`) went offline after
the event, so `submit.py --compiler compile.py` can no longer get a real
pass/fail against a GPU. This note records how `compile.py` was checked
against the **published hidden evaluation set** without one, and the result.

## The hidden set is public now

After the event, the organizers published the complete hidden set — unchanged
— at `cloudrift-ai/emmy-hackathon-2026`'s `problems/hidden/`. Per that repo's
`problems/PROVENANCE.md`:

> "The 11 problems in `hidden/` are the hidden evaluation set, published
> unchanged after the event."

Not a sample — this is the exact set every submission (including this one)
was graded against. It's included in this repo at [`problems/hidden/`](problems/hidden/).

## How it was checked without a GPU

`tools/generate.py` (in the upstream repo) shows the hidden set's inputs are
fully deterministic:

- Each `InputOp` argument is `np.random.default_rng(seed).standard_normal(shape)`,
  where `seed` is the argument's own index — recorded per-arg in `meta.json`.
- Each `ConstantOp` argument carries its fixed value directly in `meta.json`
  (`"data": [value]`).

That means the exact inputs `expected.npz` was computed from can be
rebuilt without needing the original runner or GPU.

[`tools/verify_hidden.py`](tools/verify_hidden.py) is an independent numpy
interpreter of the same lowered Kernel IR `compile.py` consumes — `Tile`,
`StridedLoop`, `Loop`, `WarpShuffle`, `TreeHalve`, `Load`, `Write`, `Assign`,
`Accum`, `Cond`, `Smem`, `CpAsync*` — written fresh in plain python/numpy
arithmetic, not by reusing `compile.py`'s own PTX-emission code. It rebuilds
each hidden problem's exact inputs, runs them through that interpreter, and
compares against `expected.npz`.

**What this checks:** whether `compile.py`'s IR-walking logic — which ops it
runs, which indices it computes, which accumulate pattern it uses — produces
the right answer for every hidden problem.

**What this does *not* check:** PTX-instruction-level correctness (a typo in
`compile.py`'s emitted text that `ptxas` would reject or a real GPU would
execute differently). See the next section for that.

## Result (numpy check)

Run it yourself:

```bash
pip install emmy-ml==0.3.11 numpy requests
python tools/verify_hidden.py           # every hidden problem
python tools/verify_hidden.py h_p1_sub  # one problem by name
```

| Problem | Category | Max abs err |
|---|---|---|
| `h_p1_sub` | pointwise | 1.79e-07 |
| `h_p2_abs_scale` | pointwise | 0.0 |
| `h_p3_sigmoid` | pointwise | 2.97e-08 |
| `h_r1_sum_4096` | reductions | 1.92e-06 |
| `h_r2_row_mean` | reductions | 1.60e-09 |
| `h_r3_sq_sum` | reductions | 3.76e-04 (tol 1e-3) |
| `h_m1_matmul_256` | matmul | 1.99e-06 |
| `h_m2_matmul_tall` | matmul | 1.90e-06 |
| `h_f1_layerscale` | fused | 2.47e-07 |
| `h_f2_softmax_512` | fused | 1.70e-09 |
| `h_a1_attention_96` | attention | 3.56e-08 |

**11/11 pass**, across all five categories, with errors consistent with
ordinary floating-point summation-order differences — comfortably inside
each problem's tolerance.

## Real-GPU verification

[`tools/verify_gpu.py`](tools/verify_gpu.py) goes one step further: it runs
`compile.py`'s **actual PTX text** — not a reinterpretation of the IR —
through `pycuda.driver.module_from_buffer`, which loads it via the CUDA
driver's own JIT compiler (the same compilation step `ptxas` performs), then
launches it on a real GPU and compares the result to `expected.npz`. This is
the closest available stand-in for the hackathon's own `ptxas` + GPU pipeline
now that the hosted runner is gone.

Run (on any CUDA-capable machine):

```bash
pip install -q pycuda emmy-ml==0.3.11
python tools/verify_gpu.py
```

Developed and run on a free **Google Colab T4** (`sm_75`). One real
limitation surfaced here: `compile.py` targets `.target sm_89`, matching the
hackathon's own problems (their real runner was an RTX 5090, `sm_120` —
*newer*, so `sm_89` PTX runs on it unchanged via forward compatibility). A
T4 is *older* than `sm_89`, so its driver refuses to JIT that PTX directly.
For problems that don't use `cp.async` (an Ampere/`sm_80`+-only instruction),
the script lowers the PTX's declared target to the device's own compute
capability before loading it — same instructions, just a lower declared ISA
floor — and runs those for real. The two matmul problems that do use
`cp.async` for shared-memory tiling are skipped on this device and reported
as skipped, not silently passed or miscounted as failures, since no PTX edit
makes a missing hardware unit appear.

| Problem | Category | Result |
|---|---|---|
| `h_p1_sub` | pointwise | **GPU-PASS**, err 0.0 |
| `h_p2_abs_scale` | pointwise | **GPU-PASS**, err 0.0 |
| `h_p3_sigmoid` | pointwise | **GPU-PASS**, err 5.96e-08 |
| `h_r1_sum_4096` | reductions | **GPU-PASS**, err 0.0 |
| `h_r2_row_mean` | reductions | **GPU-PASS**, err 7.45e-09 |
| `h_r3_sq_sum` | reductions | **GPU-PASS**, err 9.77e-04 (tol 1e-3) |
| `h_m1_matmul_256` | matmul | skipped — needs `sm_80`+ for `cp.async` |
| `h_m2_matmul_tall` | matmul | skipped — needs `sm_80`+ for `cp.async` |
| `h_f1_layerscale` | fused | **GPU-PASS**, err 9.54e-07 |
| `h_f2_softmax_512` | fused | **GPU-PASS**, err 7.45e-09 |
| `h_a1_attention_96` | attention | **GPU-PASS**, err 2.98e-07 |

**9 passed for real on GPU hardware, 0 failed, 2 skipped** (device too old
for `cp.async`, not a correctness issue). The 2 skipped problems are the
same ones in the numpy check above, which both passed there, and exercise
the same `cp.async` / `_do_cpasync_copy` code path already GPU-validated
through the real `ptxas` + runner pipeline on the open set's `m01`/`m02`
problems before the hosted runner went offline. Combined, every one of the
11 hidden problems has now been checked either on real GPU hardware or
through the code path that real hardware already validated.
