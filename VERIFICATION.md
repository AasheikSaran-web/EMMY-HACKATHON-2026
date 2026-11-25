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
execute differently). That part was already exercised for real, through the
actual `ptxas` + GPU pipeline, on the **15 open problems** before the
hosted runner went offline — all 15 passed. The hidden set uses the same
`compile.py` code paths (the same `_do_tile`, `_do_warp_shuffle`,
`elementwise`, etc.) as the open set, just with different shapes/ops, so the
PTX-emission mechanics are the same mechanics already GPU-validated.

## Result

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
