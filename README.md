# EMMY-HACKATHON-2026
EMMY HACKATHON 2026

compile.py reads Emmy's Kernel IR and emits PTX directly, 

## Run

    python compile.py path/to/kernel.ir.json > out.ptx

## Test

    pip install emmy-ml==0.3.11 numpy requests
    python submit.py <problem_name> --compiler compile.py

## Structure

No dependencies beyond emmy-ml (for IR deserialization).

- `Emitter` walks the IR body tree and emits PTX instructions directly — one
  pass, no intermediate representation.
- Virtual registers (`%f`, `%r`, `%rd`, `%p`) are allocated on the fly;
  `ptxas` does physical register allocation.
- Covers: pointwise ops, reductions (`StridedLoop`/`Loop`/`Accum`), shared
  memory tiles + `cp.async`, warp shuffles, and the matmul/attention fused
  patterns — the IR node kinds in `TASK.md` that aren't marked `*`.


