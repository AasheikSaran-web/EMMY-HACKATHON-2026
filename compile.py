#!/usr/bin/env python3

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

from emmy.compiler.graph import Graph
from emmy.compiler.ir.expr import (
    BinaryExpr,
    CastExpr,
    Expr,
    FuncCallExpr,
    Literal,
    TernaryExpr,
    Var,
)
from emmy.compiler.ir.kernel.ir import (
    CpAsyncCommit,
    CpAsyncCopy,
    CpAsyncWait,
    KernelOp,
    Smem,
    Sync,
    Tile,
    TreeHalve,
    WarpShuffle,
)
from emmy.compiler.ir.stmt.blocks import Cond, Loop, StridedLoop
from emmy.compiler.ir.stmt.leaves import Accum, Assign, Init, Let, Load, Write


# Meta
def load_meta(ir_path: Path, kernel_name: str) -> dict:
    meta_path = ir_path.parent / "meta.json"
    meta = json.loads(meta_path.read_text())
    for k in meta["kernels"]:
        if k["entry"] == kernel_name:
            return k
    if len(meta["kernels"]) == 1:
        return meta["kernels"][0]
    raise ValueError(f"no meta.json kernel entry named {kernel_name!r}")


# Dtypes
_DTYPE_BYTES = {"f32": 4, "f16": 2, "f64": 8, "i32": 4, "i64": 8, "u32": 4, "u64": 8}


def dtype_bytes(dtype: str) -> int:
    return _DTYPE_BYTES.get(dtype, 4)


_SMEM_DTYPE_BYTES = {"float": 4, "double": 8, "int": 4, "__half": 2, "unsigned long long": 8}


def smem_dtype_bytes(dtype: str) -> int:
    return _SMEM_DTYPE_BYTES.get(dtype, 4)


# Registers
class Regs:
    def __init__(self) -> None:
        self.counts = {"f": 0, "r": 0, "rd": 0, "p": 0}

    def new(self, cls: str) -> str:
        self.counts[cls] += 1
        return f"%{cls}{self.counts[cls]}"


# Types
Val = tuple[str, str]


class CompileError(Exception):
    pass


# Emitter
class Emitter:
    def __init__(self, kernel: KernelOp, meta: dict) -> None:
        self.kernel = kernel
        self.meta = meta
        self.regs = Regs()
        self.lines: list[str] = []
        self.symtab: dict[str, Val] = {}
        self.label_n = 0
        self.buffer_regs: dict[str, str] = {}
        self.shapes: dict[str, tuple[int, ...]] = {}
        self.buf_dtype_bytes: dict[str, int] = {}
        for a in meta["args"]:
            self.shapes[a["name"]] = tuple(a.get("shape", ()))
            self.buf_dtype_bytes[a["name"]] = dtype_bytes(a.get("dtype", "f32"))
        self.smem_shapes: dict[str, tuple[int, ...]] = {}
        self.smem_dtype_bytes: dict[str, int] = {}
        self.smem_base_symbol: dict[str, str] = {}
        self.smem_extra_offset: dict[str, int] = {}
        self._special: dict[str, str] = {}


    # Helpers
    def emit(self, text: str) -> None:
        self.lines.append(f"\t{text}")

    def label(self, prefix: str) -> str:
        self.label_n += 1
        return f"{prefix}{self.label_n}"

    def special(self, name: str) -> str:
        if name not in self._special:
            r = self.regs.new("r")
            self.emit(f"mov.u32 \t{r}, {name};")
            self._special[name] = r
        return self._special[name]


    # Literals
    @staticmethod
    def hex_f32(value: float) -> str:
        bits = struct.unpack(">I", struct.pack(">f", float(value)))[0]
        return f"0f{bits:08X}"


    # Expressions
    def compile_expr(self, e: Expr) -> Val:
        if isinstance(e, Var):
            if e.name not in self.symtab:
                raise CompileError(f"undefined name {e.name!r}")
            return self.symtab[e.name]
        if isinstance(e, Literal):
            if e.dtype == "int" or (isinstance(e.value, int) and e.dtype != "float"):
                r = self.regs.new("r")
                self.emit(f"mov.s32 \t{r}, {int(e.value)};")
                return (r, "s32")
            f = self.regs.new("f")
            self.emit(f"mov.f32 \t{f}, {self.hex_f32(e.value)};")
            return (f, "f32")
        if isinstance(e, BinaryExpr):
            return self._compile_binary(e)
        if isinstance(e, TernaryExpr):
            p = self.compile_pred(e.cond)
            tv = self.compile_expr(e.if_true)
            fv = self.compile_expr(e.if_false)
            ty = tv[1] if tv[1] == fv[1] else "f32"
            tv = self._to(tv, ty)
            fv = self._to(fv, ty)
            dst = self.regs.new("f" if ty == "f32" else "r")
            self.emit(f"selp.{ty} \t{dst}, {tv[0]}, {fv[0]}, {p};")
            return (dst, ty)
        if isinstance(e, FuncCallExpr):
            return self._compile_funccall(e.name, [self.compile_expr(a) for a in e.args])
        if isinstance(e, CastExpr):
            v = self.compile_expr(e.expr)
            return self._cast(v, e.dtype)
        raise CompileError(f"unsupported expression node: {type(e).__name__}")

    def _to(self, v: Val, ty: str) -> Val:
        if v[1] == ty:
            return v
        return self._cast(v, "float" if ty == "f32" else "int")

    def _cast(self, v: Val, dtype: str) -> Val:
        reg, ty = v
        want_f = dtype in ("float", "f32")
        if want_f and ty == "f32":
            return v
        if not want_f and ty == "s32":
            return v
        if want_f and ty == "s32":
            dst = self.regs.new("f")
            self.emit(f"cvt.rn.f32.s32 \t{dst}, {reg};")
            return (dst, "f32")
        if not want_f and ty == "f32":
            dst = self.regs.new("r")
            self.emit(f"cvt.rzi.s32.f32 \t{dst}, {reg};")
            return (dst, "s32")
        return v

    def _arith_result_reg(self, ty: str) -> str:
        return self.regs.new("f") if ty == "f32" else self.regs.new("r")

    _CMP_OPS = {"==": "eq", "!=": "ne", ">": "gt", "<": "lt", ">=": "ge", "<=": "le"}

    def _compile_binary(self, e: BinaryExpr) -> Val:
        if e.op in self._CMP_OPS or e.op in ("&&", "||"):
            p = self.compile_pred(e)
            dst = self.regs.new("r")
            one = self.regs.new("r")
            zero = self.regs.new("r")
            self.emit(f"mov.s32 \t{one}, 1;")
            self.emit(f"mov.s32 \t{zero}, 0;")
            self.emit(f"selp.s32 \t{dst}, {one}, {zero}, {p};")
            return (dst, "s32")
        left = self.compile_expr(e.left)
        right = self.compile_expr(e.right)
        is_f = left[1] == "f32" or right[1] == "f32"
        ty = "f32" if is_f else "s32"
        left = self._to(left, ty)
        right = self._to(right, ty)
        dst = self._arith_result_reg(ty)
        if e.op == "+":
            self.emit(f"add.{ty} \t{dst}, {left[0]}, {right[0]};")
        elif e.op == "-":
            self.emit(f"sub.{ty} \t{dst}, {left[0]}, {right[0]};")
        elif e.op == "*":
            self.emit(f"mul.{'f32' if is_f else 'lo.s32'} \t{dst}, {left[0]}, {right[0]};")
        elif e.op in ("/", "//"):
            if is_f:
                self.emit(f"div.rn.f32 \t{dst}, {left[0]}, {right[0]};")
            else:
                self.emit(f"div.s32 \t{dst}, {left[0]}, {right[0]};")
        elif e.op == "%":
            self.emit(f"rem.s32 \t{dst}, {left[0]}, {right[0]};")
        else:
            raise CompileError(f"unsupported binary op {e.op!r}")
        return (dst, ty)

    def compile_pred(self, e: Expr) -> str:
        if isinstance(e, BinaryExpr) and e.op in self._CMP_OPS:
            left = self.compile_expr(e.left)
            right = self.compile_expr(e.right)
            is_f = left[1] == "f32" or right[1] == "f32"
            ty = "f32" if is_f else "s32"
            left = self._to(left, ty)
            right = self._to(right, ty)
            p = self.regs.new("p")
            self.emit(f"setp.{self._CMP_OPS[e.op]}.{ty} \t{p}, {left[0]}, {right[0]};")
            return p
        if isinstance(e, BinaryExpr) and e.op in ("&&", "||"):
            lp = self.compile_pred(e.left)
            rp = self.compile_pred(e.right)
            p = self.regs.new("p")
            op = "and" if e.op == "&&" else "or"
            self.emit(f"{op}.pred \t{p}, {lp}, {rp};")
            return p
        v = self.compile_expr(e)
        p = self.regs.new("p")
        zero = "0" if v[1] != "f32" else "0f00000000"
        self.emit(f"setp.ne.{v[1]} \t{p}, {v[0]}, {zero};")
        return p

    _LOG2E = 1.4426950408889634

    def _compile_funccall(self, name: str, args: list[Val]) -> Val:
        if name in ("exp", "exp_fast"):
            x = self._to(args[0], "f32")
            scaled = self.regs.new("f")
            self.emit(f"mul.f32 \t{scaled}, {x[0]}, {self.hex_f32(self._LOG2E)};")
            dst = self.regs.new("f")
            self.emit(f"ex2.approx.f32 \t{dst}, {scaled};")
            return (dst, "f32")
        if name == "sqrt":
            x = self._to(args[0], "f32")
            dst = self.regs.new("f")
            self.emit(f"sqrt.rn.f32 \t{dst}, {x[0]};")
            return (dst, "f32")
        if name == "rsqrt":
            x = self._to(args[0], "f32")
            dst = self.regs.new("f")
            self.emit(f"rsqrt.approx.f32 \t{dst}, {x[0]};")
            return (dst, "f32")
        if name == "fmax":
            a, b = self._to(args[0], "f32"), self._to(args[1], "f32")
            dst = self.regs.new("f")
            self.emit(f"max.f32 \t{dst}, {a[0]}, {b[0]};")
            return (dst, "f32")
        if name == "fmin":
            a, b = self._to(args[0], "f32"), self._to(args[1], "f32")
            dst = self.regs.new("f")
            self.emit(f"min.f32 \t{dst}, {a[0]}, {b[0]};")
            return (dst, "f32")
        if name == "fabs":
            x = self._to(args[0], "f32")
            dst = self.regs.new("f")
            self.emit(f"abs.f32 \t{dst}, {x[0]};")
            return (dst, "f32")
        if name == "log":
            x = self._to(args[0], "f32")
            l2 = self.regs.new("f")
            self.emit(f"lg2.approx.f32 \t{l2}, {x[0]};")
            dst = self.regs.new("f")
            self.emit(f"mul.f32 \t{dst}, {l2}, {self.hex_f32(1.0 / self._LOG2E)};")
            return (dst, "f32")
        raise CompileError(f"unsupported intrinsic {name!r}")


    # Elementwise
    def elementwise(self, op_name: str, vals: list[Val]) -> Val:
        fvals = [self._to(v, "f32") for v in vals]
        if op_name == "add":
            dst = self.regs.new("f")
            self.emit(f"add.f32 \t{dst}, {fvals[0][0]}, {fvals[1][0]};")
            return (dst, "f32")
        if op_name == "subtract":
            dst = self.regs.new("f")
            self.emit(f"sub.f32 \t{dst}, {fvals[0][0]}, {fvals[1][0]};")
            return (dst, "f32")
        if op_name == "multiply":
            dst = self.regs.new("f")
            self.emit(f"mul.f32 \t{dst}, {fvals[0][0]}, {fvals[1][0]};")
            return (dst, "f32")
        if op_name == "divide":
            dst = self.regs.new("f")
            self.emit(f"div.rn.f32 \t{dst}, {fvals[0][0]}, {fvals[1][0]};")
            return (dst, "f32")
        if op_name == "maximum":
            dst = self.regs.new("f")
            self.emit(f"max.f32 \t{dst}, {fvals[0][0]}, {fvals[1][0]};")
            return (dst, "f32")
        if op_name == "minimum":
            dst = self.regs.new("f")
            self.emit(f"min.f32 \t{dst}, {fvals[0][0]}, {fvals[1][0]};")
            return (dst, "f32")
        if op_name == "reciprocal":
            one = self.regs.new("f")
            self.emit(f"mov.f32 \t{one}, {self.hex_f32(1.0)};")
            dst = self.regs.new("f")
            self.emit(f"div.rn.f32 \t{dst}, {one}, {fvals[0][0]};")
            return (dst, "f32")
        if op_name == "negative":
            dst = self.regs.new("f")
            self.emit(f"neg.f32 \t{dst}, {fvals[0][0]};")
            return (dst, "f32")
        if op_name == "square":
            dst = self.regs.new("f")
            self.emit(f"mul.f32 \t{dst}, {fvals[0][0]}, {fvals[0][0]};")
            return (dst, "f32")
        if op_name == "relu":
            zero = self.regs.new("f")
            self.emit(f"mov.f32 \t{zero}, {self.hex_f32(0.0)};")
            dst = self.regs.new("f")
            self.emit(f"max.f32 \t{dst}, {zero}, {fvals[0][0]};")
            return (dst, "f32")
        if op_name == "abs":
            dst = self.regs.new("f")
            self.emit(f"abs.f32 \t{dst}, {fvals[0][0]};")
            return (dst, "f32")
        if op_name == "sigmoid":
            neg = self.regs.new("f")
            self.emit(f"neg.f32 \t{neg}, {fvals[0][0]};")
            e = self._compile_funccall("exp", [(neg, "f32")])
            one = self.regs.new("f")
            self.emit(f"mov.f32 \t{one}, {self.hex_f32(1.0)};")
            denom = self.regs.new("f")
            self.emit(f"add.f32 \t{denom}, {one}, {e[0]};")
            dst = self.regs.new("f")
            self.emit(f"div.rn.f32 \t{dst}, {one}, {denom};")
            return (dst, "f32")
        if op_name == "copy":
            return fvals[0]
        if op_name in ("exp", "exp_fast", "sqrt", "rsqrt", "log"):
            return self._compile_funccall(op_name, fvals)
        if op_name == "where":
            p = self.regs.new("p")
            zero = self.regs.new("f")
            self.emit(f"mov.f32 \t{zero}, {self.hex_f32(0.0)};")
            self.emit(f"setp.ne.f32 \t{p}, {fvals[0][0]}, {zero};")
            dst = self.regs.new("f")
            self.emit(f"selp.f32 \t{dst}, {fvals[1][0]}, {fvals[2][0]}, {p};")
            return (dst, "f32")
        raise CompileError(f"unsupported elementwise op {op_name!r}")


    # Indexing
    def flatten_index(self, buf: str, index: tuple[Expr, ...]) -> Expr:
        if len(index) <= 1:
            return index[0] if index else Literal(0, "int")
        shape = self.shapes.get(buf) or self.smem_shapes.get(buf)
        if not shape or len(shape) != len(index):
            flat: Expr = index[0]
            for i in index[1:]:
                flat = BinaryExpr("+", flat, i)
            return flat
        flat = None
        for d, idx in enumerate(index):
            stride = 1
            for k in range(d + 1, len(shape)):
                stride *= shape[k]
            term: Expr = idx if stride == 1 else BinaryExpr("*", idx, Literal(stride, "int"))
            flat = term if flat is None else BinaryExpr("+", flat, term)
        return flat

    @staticmethod
    def _const_eval(e: Expr) -> int | None:
        if isinstance(e, Literal):
            return int(e.value)
        if isinstance(e, BinaryExpr):
            l = Emitter._const_eval(e.left)
            r = Emitter._const_eval(e.right)
            if l is None or r is None:
                return None
            if e.op == "+":
                return l + r
            if e.op == "-":
                return l - r
            if e.op == "*":
                return l * r
        return None

    def global_addr(self, buf: str, index: tuple[Expr, ...]) -> tuple[str, int]:
        return self.global_addr_flat(buf, self.flatten_index(buf, index))

    def global_addr_flat(self, buf: str, flat: Expr) -> tuple[str, int]:
        esz = self.buf_dtype_bytes.get(buf, 4)
        base = self.buffer_regs[buf]
        c = self._const_eval(flat)
        if c is not None:
            return base, c * esz
        val = self.compile_expr(flat)
        val = self._to(val, "s32")
        tmp = self.regs.new("rd")
        self.emit(f"mul.wide.s32 \t{tmp}, {val[0]}, {esz};")
        addr = self.regs.new("rd")
        self.emit(f"add.s64 \t{addr}, {base}, {tmp};")
        return addr, 0

    def smem_addr(self, name: str, index: tuple[Expr, ...]) -> tuple[str, int]:
        return self.smem_addr_flat(name, self.flatten_index(name, index))

    def smem_addr_flat(self, name: str, flat: Expr) -> tuple[str, int]:
        esz = self.smem_dtype_bytes.get(name, 4)
        sym = self.smem_base_symbol.get(name, name)
        extra = self.smem_extra_offset.get(name, 0)
        c = self._const_eval(flat)
        if c is not None:
            return sym, extra + c * esz
        val = self._to(self.compile_expr(flat), "s32")
        base = self.regs.new("r")
        self.emit(f"mov.u32 \t{base}, {sym};")
        off = self.regs.new("r")
        self.emit(f"mul.lo.s32 \t{off}, {val[0]}, {esz};")
        if extra:
            off2 = self.regs.new("r")
            self.emit(f"add.s32 \t{off2}, {off}, {extra};")
            off = off2
        addr = self.regs.new("r")
        self.emit(f"add.s32 \t{addr}, {base}, {off};")
        return addr, 0

    @staticmethod
    def addr_operand(base: str, off: int) -> str:
        return f"[{base}+{off}]" if off else f"[{base}]"


    # Dispatch
    def compile_body(self, body) -> None:
        for s in body:
            self.compile_stmt(s)

    def compile_stmt(self, s) -> None:
        if isinstance(s, Tile):
            self._do_tile(s)
        elif isinstance(s, Smem):
            self._do_smem(s)
        elif isinstance(s, Sync):
            self._do_sync(s)
        elif isinstance(s, Load):
            self._do_load(s)
        elif isinstance(s, Write):
            self._do_write(s)
        elif isinstance(s, Assign):
            self._do_assign(s)
        elif isinstance(s, Accum):
            self._do_accum(s)
        elif isinstance(s, Let):
            self._do_let(s)
        elif isinstance(s, Init):
            self._do_init(s)
        elif isinstance(s, Cond):
            self._do_cond(s)
        elif isinstance(s, StridedLoop):
            self._do_strided_loop(s)
        elif isinstance(s, Loop):
            self._do_loop(s)
        elif isinstance(s, WarpShuffle):
            self._do_warp_shuffle(s)
        elif isinstance(s, TreeHalve):
            self._do_tree_halve(s)
        elif isinstance(s, CpAsyncCopy):
            self._do_cpasync_copy(s)
        elif isinstance(s, CpAsyncCommit):
            self.emit("cp.async.commit_group;")
        elif isinstance(s, CpAsyncWait):
            self.emit(f"cp.async.wait_group {s.group};")
        else:
            raise CompileError(f"unsupported statement kind: {type(s).__name__}")


    # Tile
    def _do_tile(self, s: Tile) -> None:
        self.emit("// Tile: thread -> axis-coordinate decode + bounds guard")
        tid = self.special("%tid.x")
        ctaid = self.special("%ctaid.x")
        ntid = self.special("%ntid.x")
        gid = self.regs.new("r")
        if s.aux_threads and s.block_threads is not None:
            tmp = self.regs.new("r")
            self.emit(f"mul.lo.s32 \t{tmp}, {ctaid}, {s.block_threads};")
            modded = self.regs.new("r")
            bt = self.regs.new("r")
            self.emit(f"mov.s32 \t{bt}, {s.block_threads};")
            self.emit(f"rem.s32 \t{modded}, {tid}, {bt};")
            self.emit(f"add.s32 \t{gid}, {tmp}, {modded};")
        else:
            tmp = self.regs.new("r")
            self.emit(f"mul.lo.s32 \t{tmp}, {ctaid}, {ntid};")
            self.emit(f"add.s32 \t{gid}, {tmp}, {tid};")

        extents = [int(a.extent.as_static()) for a in s.axes]
        n = 1
        for e in extents:
            n *= e
        n_lit = self.regs.new("r")
        self.emit(f"mov.s32 \t{n_lit}, {n};")
        p = self.regs.new("p")
        self.emit(f"setp.ge.s32 \t{p}, {gid}, {n_lit};")
        skip = self.label("TILE_SKIP_")
        self.emit(f"@{p} bra \t{skip};")

        strides = [1] * len(extents)
        acc = 1
        for i in range(len(extents) - 1, -1, -1):
            strides[i] = acc
            acc *= extents[i]
        for i, axis in enumerate(s.axes):
            stride, extent = strides[i], extents[i]
            if len(s.axes) == 1:
                self.symtab[axis.name] = (gid, "s32")
                continue
            if stride == 1:
                v = self.regs.new("r")
                self.emit(f"rem.s32 \t{v}, {gid}, {self._const32(extent)};")
                self.symtab[axis.name] = (v, "s32")
            elif i == 0:
                v = self.regs.new("r")
                self.emit(f"div.s32 \t{v}, {gid}, {self._const32(stride)};")
                self.symtab[axis.name] = (v, "s32")
            else:
                d = self.regs.new("r")
                self.emit(f"div.s32 \t{d}, {gid}, {self._const32(stride)};")
                v = self.regs.new("r")
                self.emit(f"rem.s32 \t{v}, {d}, {self._const32(extent)};")
                self.symtab[axis.name] = (v, "s32")

        self.compile_body(s.body)
        self.emit(f"{skip}:")

    def _const32(self, value: int) -> str:
        r = self.regs.new("r")
        self.emit(f"mov.s32 \t{r}, {value};")
        return r


    # Smem
    def _do_smem(self, s: Smem) -> None:
        self.smem_shapes[s.name] = tuple(int(e) for e in s.extents)
        self.smem_dtype_bytes[s.name] = smem_dtype_bytes(s.dtype)

    def _do_sync(self, s: Sync) -> None:
        if s.warp:
            self.emit("bar.warp.sync \t-1;")
        elif s.barrier_id == 0:
            self.emit("bar.sync \t0;")
        else:
            self.emit(f"bar.sync \t{s.barrier_id}, {s.count};")


    # Transfer
    def _is_smem(self, name: str) -> bool:
        return name in self.smem_shapes

    def _do_load(self, s: Load) -> None:
        n = s.width
        is_smem = self._is_smem(s.input)
        space = "shared" if is_smem else "global"
        if is_smem:
            base, off = self.smem_addr(s.input, s.index)
            esz = self.smem_dtype_bytes.get(s.input, 4)
        else:
            base, off = self.global_addr(s.input, s.index)
            esz = self.buf_dtype_bytes.get(s.input, 4)
        if n == 1:
            dst = self.regs.new("f")
            self.emit(f"ld.{space}.f32 \t{dst}, {self.addr_operand(base, off)};")
            self.symtab[s.names[0]] = (dst, "f32")
            return
        if n in (2, 4):
            dsts = [self.regs.new("f") for _ in range(n)]
            self.emit(f"ld.{space}.v{n}.f32 \t{{{', '.join(dsts)}}}, {self.addr_operand(base, off)};")
            for nm, d in zip(s.names, dsts, strict=True):
                self.symtab[nm] = (d, "f32")
            return
        for k, nm in enumerate(s.names):
            dst = self.regs.new("f")
            self.emit(f"ld.{space}.f32 \t{dst}, {self.addr_operand(base, off + k * esz)};")
            self.symtab[nm] = (dst, "f32")

    def _do_write(self, s: Write) -> None:
        n = s.width
        is_smem = self._is_smem(s.output)
        space = "shared" if is_smem else "global"
        if is_smem:
            base, off = self.smem_addr(s.output, s.index)
            esz = self.smem_dtype_bytes.get(s.output, 4)
        else:
            base, off = self.global_addr(s.output, s.index)
            esz = self.buf_dtype_bytes.get(s.output, 4)
        if n == 1:
            val = self._to(self.symtab[s.values[0]], "f32")
            if s.atomic:
                self.emit(f"red.{space}.add.f32 \t{self.addr_operand(base, off)}, {val[0]};")
            else:
                self.emit(f"st.{space}.f32 \t{self.addr_operand(base, off)}, {val[0]};")
            return
        vals = [self._to(self.symtab[v], "f32")[0] for v in s.values]
        if n in (2, 4):
            self.emit(f"st.{space}.v{n}.f32 \t{self.addr_operand(base, off)}, {{{', '.join(vals)}}};")
            return
        for k, v in enumerate(vals):
            self.emit(f"st.{space}.f32 \t{self.addr_operand(base, off + k * esz)}, {v};")

    def _do_cpasync_copy(self, s: CpAsyncCopy) -> None:
        smem_flat = self.flatten_index(s.smem, s.smem_index)
        if s.lane_index is not None:
            cols = self.smem_shapes.get(s.smem, (0, 0))[-1]
            lane_flat: Expr = BinaryExpr(
                "+", BinaryExpr("*", s.lane_index[0], Literal(cols, "int")), s.lane_index[1]
            )
            smem_flat = BinaryExpr("+", smem_flat, lane_flat)
        src_flat = self.flatten_index(s.src, s.src_index)
        smem_base, smem_off = self.smem_addr_flat(s.smem, smem_flat)
        glob_base, glob_off = self.global_addr_flat(s.src, src_flat)
        qual = "cg" if s.nbytes == 16 else "ca"
        self.emit(
            f"cp.async.{qual}.shared.global \t{self.addr_operand(smem_base, smem_off)}, "
            f"{self.addr_operand(glob_base, glob_off)}, {s.nbytes};"
        )


    # Assign
    def _do_assign(self, s: Assign) -> None:
        vals = [self.symtab[a] for a in s.args]
        self.symtab[s.name] = self.elementwise(s.op.name, vals)

    _ACCUM_OP = {"add": "add", "maximum": "max", "minimum": "min", "multiply": "mul"}

    def _do_accum(self, s: Accum) -> None:
        acc = self.symtab[s.name]
        left = self.symtab[s.base] if s.base is not None else acc
        val = self._to(self.symtab[s.value], "f32")
        op = self._ACCUM_OP.get(s.op.name, "add")
        self.emit(f"{op}.f32 \t{acc[0]}, {left[0]}, {val[0]};")
        self.symtab[s.name] = acc

    def _do_let(self, s: Let) -> None:
        self.symtab[s.name] = self.compile_expr(s.value)

    def _do_init(self, s: Init) -> None:
        f = self.regs.new("f")
        self.emit(f"mov.f32 \t{f}, {self.hex_f32(s.identity)};")
        self.symtab[s.name] = (f, "f32")


    # Cond
    def _do_cond(self, s: Cond) -> None:
        p = self.compile_pred(s.cond)
        if not s.else_body:
            skip = self.label("COND_SKIP_")
            self.emit(f"@!{p} bra \t{skip};")
            self.compile_body(s.body)
            self.emit(f"{skip}:")
            return
        else_l = self.label("COND_ELSE_")
        end_l = self.label("COND_END_")
        self.emit(f"@!{p} bra \t{else_l};")
        self.compile_body(s.body)
        self.emit(f"bra \t{end_l};")
        self.emit(f"{else_l}:")
        self.compile_body(s.else_body)
        self.emit(f"{end_l}:")


    # Loops
    def _seed_accums(self, body) -> None:
        seen: set[str] = set()
        for s in body:
            if isinstance(s, Accum) and s.name not in seen:
                seen.add(s.name)
                f = self.regs.new("f")
                ident = s.op.identity if s.op.identity is not None else 0.0
                self.emit(f"mov.f32 \t{f}, {self.hex_f32(ident)};")
                self.symtab[s.name] = (f, "f32")

    def _loop_bound(self, e) -> Val:
        return self._to(self.compile_expr(e), "s32")

    def _do_strided_loop(self, s: StridedLoop) -> None:
        self.emit(f"// StridedLoop {s.axis.name}")
        if s.seed:
            self._seed_accums(s.body)
        start = self._to(self.compile_expr(s.start), "s32")
        var = self.regs.new("r")
        self.emit(f"mov.s32 \t{var}, {start[0]};")
        self.symtab[s.axis.name] = (var, "s32")
        bound_expr = s.end if s.end is not None else Literal(int(s.axis.extent.as_static()), "int")
        step = self._to(self.compile_expr(s.step) if isinstance(s.step, Expr) else (self._const32(int(s.step)), "s32"), "s32")

        top = self.label("SLOOP_")
        end = self.label("SLOOP_END_")
        self.emit(f"{top}:")
        bound = self._loop_bound(bound_expr)
        p = self.regs.new("p")
        self.emit(f"setp.ge.s32 \t{p}, {var}, {bound[0]};")
        self.emit(f"@{p} bra \t{end};")
        self.compile_body(s.body)
        self.emit(f"add.s32 \t{var}, {var}, {step[0]};")
        self.emit(f"bra \t{top};")
        self.emit(f"{end}:")

    def _do_loop(self, s: Loop) -> None:
        self.emit(f"// Loop {s.axis.name}")
        if s.seed:
            self._seed_accums(s.body)
        var = self.regs.new("r")
        self.emit(f"mov.s32 \t{var}, 0;")
        self.symtab[s.axis.name] = (var, "s32")
        n = int(s.axis.extent.as_static())
        top = self.label("LOOP_")
        end = self.label("LOOP_END_")
        self.emit(f"{top}:")
        p = self.regs.new("p")
        self.emit(f"setp.ge.s32 \t{p}, {var}, {self._const32(n)};")
        self.emit(f"@{p} bra \t{end};")
        self.compile_body(s.body)
        self.emit(f"add.s32 \t{var}, {var}, 1;")
        self.emit(f"bra \t{top};")
        self.emit(f"{end}:")


    # Shuffle
    def _run_combine(self, combine_states) -> None:
        for a in combine_states:
            vals = [self.symtab[arg] for arg in a.args]
            self.symtab[a.name] = self.elementwise(a.op.name, vals)

    def _do_warp_shuffle(self, s: WarpShuffle) -> None:
        self.emit(f"// WarpShuffle length={s.length}")
        step = s.length // 2
        while step > 0:
            for st, stb in zip(s.state, s.state_b, strict=True):
                v = self.symtab[st]
                bits = self.regs.new("r")
                if v[1] == "f32":
                    self.emit(f"mov.b32 \t{bits}, {v[0]};")
                else:
                    bits = v[0]
                shout = self.regs.new("r")
                self.emit(f"shfl.sync.bfly.b32 \t{shout}, {bits}, {step}, 0x1f, 0xffffffff;")
                if v[1] == "f32":
                    outf = self.regs.new("f")
                    self.emit(f"mov.b32 \t{outf}, {shout};")
                    self.symtab[stb] = (outf, "f32")
                else:
                    self.symtab[stb] = (shout, "s32")
            self._run_combine(s.combine_states)
            step //= 2

    def _do_tree_halve(self, s: TreeHalve) -> None:
        self.emit(f"// TreeHalve length={s.length} tid={s.tid_var}")
        tid = self.symtab[s.tid_var]
        step = s.length // 2
        while step > 0:
            p = self.regs.new("p")
            self.emit(f"setp.lt.s32 \t{p}, {tid[0]}, {step};")
            skip = self.label("TH_SKIP_")
            self.emit(f"@!{p} bra \t{skip};")
            addrs = []
            for buf, st, stb in zip(s.bufs, s.state, s.state_b, strict=True):
                esz = self.smem_dtype_bytes.get(buf, 4)
                slot_off = self.regs.new("r")
                self.emit(f"mul.lo.s32 \t{slot_off}, {tid[0]}, {esz};")
                base = self.regs.new("r")
                self.emit(f"mov.u32 \t{base}, {buf};")
                addr = self.regs.new("r")
                self.emit(f"add.s32 \t{addr}, {base}, {slot_off};")
                v1 = self.regs.new("f")
                self.emit(f"ld.shared.f32 \t{v1}, [{addr}];")
                v2 = self.regs.new("f")
                self.emit(f"ld.shared.f32 \t{v2}, [{addr}+{step * esz}];")
                self.symtab[st] = (v1, "f32")
                self.symtab[stb] = (v2, "f32")
                addrs.append(addr)
            self._run_combine(s.combine_states)
            for addr, st in zip(addrs, s.state, strict=True):
                self.emit(f"st.shared.f32 \t[{addr}], {self.symtab[st][0]};")
            self.emit(f"{skip}:")
            self.emit("bar.sync \t0;")
            step //= 2
        for buf, st in zip(s.bufs, s.state, strict=True):
            base = self.regs.new("r")
            self.emit(f"mov.u32 \t{base}, {buf};")
            v = self.regs.new("f")
            self.emit(f"ld.shared.f32 \t{v}, [{base}];")
            self.symtab[st] = (v, "f32")


    # Warp
    def _needs_lane_warp(self, body) -> bool:
        for s in body:
            if isinstance(s, (WarpShuffle, TreeHalve)):
                return True
            for b in s.nested():
                if self._needs_lane_warp(b):
                    return True
        return False

    def _bind_lane_warp(self) -> None:
        tid = self.special("%tid.x")
        lane = self.regs.new("r")
        self.emit(f"and.b32 \t{lane}, {tid}, 31;")
        warp = self.regs.new("r")
        self.emit(f"shr.u32 \t{warp}, {tid}, 5;")
        self.symtab["lane"] = (lane, "s32")
        self.symtab["warp"] = (warp, "s32")


    # Run
    def collect_smem(self, body) -> list[Smem]:
        out: list[Smem] = []
        for s in body:
            if isinstance(s, Smem):
                out.append(s)
            for b in s.nested():
                out.extend(self.collect_smem(b))
        return out

    def run(self) -> str:
        body = self.kernel.body
        smems = self.collect_smem(body)
        sizes: list[tuple[Smem, int, int]] = []
        total_bytes = 0
        for sm in smems:
            total = 1
            for e in sm.extents:
                total *= int(e)
            nbytes = total * smem_dtype_bytes(sm.dtype)
            align = sm.align or 4
            sizes.append((sm, nbytes, align))
            self.smem_shapes[sm.name] = tuple(int(e) for e in sm.extents)
            self.smem_dtype_bytes[sm.name] = smem_dtype_bytes(sm.dtype)
            total_bytes += nbytes

        smem_decls: list[str] = []
        if total_bytes <= 49152 or not sizes:
            for sm, nbytes, align in sizes:
                smem_decls.append(f".shared .align {align} .b8 {sm.name}[{nbytes}];")
                self.smem_base_symbol[sm.name] = sm.name
                self.smem_extra_offset[sm.name] = 0
        else:
            dyn_sym = "__dyn_smem"
            max_align = max(align for _, _, align in sizes)
            offset = 0
            for sm, nbytes, align in sizes:
                if offset % align:
                    offset += align - (offset % align)
                self.smem_base_symbol[sm.name] = dyn_sym
                self.smem_extra_offset[sm.name] = offset
                offset += nbytes
            smem_decls.append(f".extern .shared .align {max_align} .b8 {dyn_sym}[];")

        arg_order = self.meta["arg_order"]
        params = [f"\t.param .u64 param_{nm}" for nm in arg_order]
        sig = f".visible .entry {self.kernel.name}(\n" + ",\n".join(params) + "\n)"

        for nm in arg_order:
            rd_in = self.regs.new("rd")
            self.emit(f"ld.param.u64 \t{rd_in}, [param_{nm}];")
            rd_g = self.regs.new("rd")
            self.emit(f"cvta.to.global.u64 \t{rd_g}, {rd_in};")
            self.buffer_regs[nm] = rd_g

        if self._needs_lane_warp(body):
            self._bind_lane_warp()

        self.compile_body(body)
        self.emit("ret;")

        decl_lines = []
        if self.regs.counts["p"]:
            decl_lines.append(f"\t.reg .pred \t%p<{self.regs.counts['p'] + 1}>;")
        if self.regs.counts["f"]:
            decl_lines.append(f"\t.reg .f32 \t%f<{self.regs.counts['f'] + 1}>;")
        if self.regs.counts["r"]:
            decl_lines.append(f"\t.reg .b32 \t%r<{self.regs.counts['r'] + 1}>;")
        if self.regs.counts["rd"]:
            decl_lines.append(f"\t.reg .b64 \t%rd<{self.regs.counts['rd'] + 1}>;")

        out = ["// generated by compile.py", ".version 8.0", ".target sm_89", ".address_size 64", ""]
        out.extend(smem_decls)
        if smem_decls:
            out.append("")
        out.append(sig)
        out.append("{")
        out.extend(decl_lines)
        out.append("")
        out.extend(self.lines)
        out.append("}")
        out.append("")
        return "\n".join(out)


# Pipeline
def find_kernel_op(graph: Graph) -> KernelOp:
    for node in graph.nodes.values():
        if isinstance(node.op, KernelOp):
            return node.op
    raise CompileError("no KernelOp found in graph")


def run_pipeline(ir_path: Path) -> str:
    data = json.loads(ir_path.read_text())
    graph = Graph.from_dict(data)
    kernel = find_kernel_op(graph)
    meta = load_meta(ir_path, kernel.name)
    return Emitter(kernel, meta).run()


# Entry
def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python compile.py path/to/<kernel>.ir.json > out.ptx", file=sys.stderr)
        sys.exit(1)
    ptx = run_pipeline(Path(sys.argv[1]))
    sys.stdout.write(ptx)


if __name__ == "__main__":
    main()
