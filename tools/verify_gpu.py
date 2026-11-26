"""Run compile.py's actual PTX output against problems/hidden/ on a real GPU.

Needs a CUDA GPU (this machine has none -- developed and run on a Colab T4).
pycuda's module_from_buffer loads the PTX through the driver's own JIT
compiler (the same compilation step ptxas performs), so this exercises the
PTX text itself, not just the IR-walking logic tools/verify_hidden.py checks.

Usage (e.g. in a Colab T4 runtime):
    pip install -q pycuda
    git clone https://github.com/AasheikSaran-web/EMMY-HACKATHON-2026 repo
    cd repo && python3 tools/verify_gpu.py
"""
import json, subprocess, sys
from pathlib import Path
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

REPO = Path(__file__).resolve().parent.parent
HIDDEN = REPO / "problems" / "hidden"


def build_inputs(k):
    arrays = {}
    for a in k["args"]:
        if a.get("output"):
            continue
        if "data" in a:
            n = 1
            for d in (a.get("shape") or [1]):
                n *= d
            arrays[a["name"]] = np.full(n, a["data"][0], dtype=np.float32)
        elif "seed" in a:
            rng = np.random.default_rng(a["seed"])
            arrays[a["name"]] = rng.standard_normal(a.get("shape") or [1]).astype(np.float32)
    return arrays


def run_problem(pdir):
    top_meta = json.loads((pdir / "meta.json").read_text())
    produced = {}
    for k in top_meta["kernels"]:
        ir_file = next(pdir.glob(f"{k['entry']}.ir.json"))
        r = subprocess.run([sys.executable, str(REPO / "compile.py"), str(ir_file)],
                            capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  COMPILE FAIL: {r.stderr}")
            return False
        ptx = r.stdout

        mod = cuda.module_from_buffer(ptx.encode())
        func = mod.get_function(k["entry"])

        inputs = build_inputs(k)
        dev_ptrs = {}
        host_arrays = {}
        for a in k["args"]:
            name = a["name"]
            shape = a.get("shape") or [1]
            n = 1
            for d in shape:
                n *= d
            if a.get("output"):
                arr = np.zeros(n, dtype=np.float32)
            elif a.get("intermediate"):
                arr = produced[name]
            else:
                arr = inputs[name]
            host_arrays[name] = arr
            dptr = cuda.mem_alloc(arr.nbytes)
            cuda.memcpy_htod(dptr, arr)
            dev_ptrs[name] = dptr

        args_in_order = [dev_ptrs[nm] for nm in k["arg_order"]]
        bx, by, bz = (list(k["block"]) + [1, 1, 1])[:3]
        gx, gy, gz = (list(k["grid"]) + [1, 1, 1])[:3]
        func(*args_in_order, block=(bx, by, bz), grid=(gx, gy, gz),
             shared=k.get("smem_bytes", 0) or 0)
        cuda.Context.synchronize()

        for a in k["args"]:
            if a.get("output"):
                out = np.empty_like(host_arrays[a["name"]])
                cuda.memcpy_dtoh(out, dev_ptrs[a["name"]])
                produced[a["name"]] = out

    expected = dict(np.load(pdir / "expected.npz"))
    rtol = top_meta.get("rtol", 1e-3)
    atol = top_meta.get("atol", 1e-5)
    ok = True
    for name, ref in expected.items():
        got = produced[name].reshape(ref.shape)
        if np.allclose(got, ref, rtol=rtol, atol=atol):
            err = float(np.max(np.abs(got - ref))) if got.size else 0.0
            print(f"  GPU-PASS {name}: max abs err {err:.3e}  (rtol={rtol} atol={atol})")
        else:
            err = float(np.max(np.abs(got - ref)))
            print(f"  GPU-FAIL {name}: max abs err {err:.3e}  (rtol={rtol} atol={atol})")
            ok = False
    return ok


def main():
    all_ok = True
    for pdir in sorted(HIDDEN.iterdir()):
        if not pdir.is_dir():
            continue
        print(f"=== {pdir.name} ===")
        try:
            ok = run_problem(pdir)
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            ok = False
        all_ok = all_ok and ok
    print()
    print("ALL PASS ON REAL GPU" if all_ok else "SOME FAILED ON REAL GPU")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
