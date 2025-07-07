#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates
import argparse
from pathlib import Path

from ml_dtypes import bfloat16
import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col1, NPU2
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorTiler2D

base_dir = Path(__file__).parent

dtype_map = {
    "bf16": bfloat16,
    "i8": np.int8,
    "i16": np.int16,
    "f32": np.float32,
    "i32": np.int32,
}

microkernel_mac_dim_map = {
    "npu": {
        "bf16": (4, 8, 4),
        "i8": (4, 8, 8),
        "i16": (4, 4, 4),
    },
    "npu2": {
        "bf16": {
            # emulate_bf16_mmul_with_bfp16
            True: (8, 8, 8),
            False: (4, 8, 8),
        },
        "i8": (8, 8, 8),
        "i16": (4, 4, 8),
    },
}

def main():
    argparser = argparse.ArgumentParser(
        prog="AIE Matrix Multiplication MLIR Design (Single Core)",
        description="Emits MLIR code for a matrix multiplication design of the given input size",
    )
    argparser.add_argument("--dev", type=str, choices=["npu", "npu2"], default="npu")
    argparser.add_argument("--heads", type=int, default=1)
    argparser.add_argument("-M", type=int, default=256)
    argparser.add_argument("-K", type=int, default=256)
    argparser.add_argument("-N", type=int, default=256)
    argparser.add_argument("-m", type=int, default=64)
    argparser.add_argument("-k", type=int, default=64)
    argparser.add_argument("-n", type=int, default=64)
    argparser.add_argument(
        "--dtype_in", type=str, choices=["bf16", "i8", "i16"], default="i16"
    )
    argparser.add_argument(
        "--dtype_out",
        type=str,
        choices=["bf16", "i8", "i16", "f32", "i32"],
        default="i32",
    )
    argparser.add_argument("--transposed_b", type=int, choices=[0, 1], default=0, 
                            help="Transpose matrix B before multiplication (0: No, 1: Yes)")
    argparser.add_argument("--b-col-maj", type=int, choices=[0, 1], default=0)
    argparser.add_argument("--emulate-bf16-mmul-with-bfp16", type=bool, default=False)
    argparser.add_argument("--trace_size", type=int, default=0)
    argparser.add_argument("--output_file_path", type=str, default = base_dir / "build" / "${heads}_${M}x${K}x${N}_${m}x${k}x${n}.mlir", help="Output file path for the generated MLIR module")
    argparser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = argparser.parse_args()
    maybe_module = batched_matmul_single_core(
        args.dev,
        args.heads,
        args.M,
        args.K,
        args.N,
        args.m,
        args.k,
        args.n,
        args.dtype_in,
        args.dtype_out,
        args.transposed_b == 1,
        args.emulate_bf16_mmul_with_bfp16,
        args.trace_size,
        args.verbose
    )
    
    output_file_path = Path(args.output_file_path)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file_path, "w") as f:
        f.write(str(maybe_module))

    if args.verbose:
        print(f"MLIR module written to {output_file_path}")

def batched_matmul_single_core(
    dev: str,
    heads: int,
    M: int,
    K: int,
    N: int,
    m: int,
    k: int,
    n: int,
    dtype_in_str: str,
    dtype_out_str: str,
    transposed_b: bool,
    emulate_bf16_mmul_with_bfp16: bool,
    trace_size: int = 0,
    verbose: bool = False,
):

    vectorized = True
    enable_tracing = True if trace_size > 0 else False

    # r, s, t are the dimensions required by the microkernel MAC instructions.
    mac_dims = microkernel_mac_dim_map[dev][dtype_in_str]
    if dev == "npu2" and dtype_in_str == "bf16":
        r, s, t = mac_dims[emulate_bf16_mmul_with_bfp16]
    else:
        r, s, t = mac_dims

    if verbose:
        print(f"Device: {dev}")
        print(f"Number of heads: {heads}")
        print(f"Matrix dimensions: M={M}, K={K}, N={N}, m={m}, k={k}, n={n}")
        print(f"Data types: Input={dtype_in_str}, Output={dtype_out_str}")
        print(f"Microkernel MAC dimensions: r={r}, s={s}, t={t}")
        print(f"Vectorized: {vectorized}")
        print(f"Enable tracing: {enable_tracing}")
        
    assert heads > 0, "Number of heads must be greater than 0"
    assert M % m == 0, f"M must be divisible by m ({M} % {m} != 0)"
    assert K % k == 0, f"K must be divisible by k ({K} % {k} != 0)"
    assert N % n == 0, f"N must be divisible by n ({N} % {n} != 0)"
    
    assert m % r == 0, f"m must be divisible by r ({m} % {r} != 0)"
    assert k % s == 0, f"k must be divisible by s ({k} % {s} != 0)"
    assert n % t == 0, f"n must be divisible by t ({n} % {t} != 0)"

    dtype_in = dtype_map[dtype_in_str]
    dtype_out = dtype_map[dtype_out_str]

    assert np.issubdtype(dtype_in, np.integer) == np.issubdtype(
        dtype_out, np.integer
    ), f"Input dtype ({dtype_in}) and output dtype ({dtype_out}) must either both be integer or both be float"
    assert (
        np.dtype(dtype_out).itemsize >= np.dtype(dtype_in).itemsize
    ), f"Output dtype ({dtype_out}) must be equal or larger to input dtype ({dtype_in})"

    M_div_m = M // m
    K_div_k = K // k
    N_div_n = N // n
    tiles = heads * M_div_m * N_div_n

    # Tensors living in DRAM
    A_ty = np.ndarray[(heads * M * K,), np.dtype[dtype_in]]
    B_ty = np.ndarray[(heads * K * N,), np.dtype[dtype_in]]
    C_ty = np.ndarray[(heads * M * N,), np.dtype[dtype_out]]
    # Tensors living in Mem Tiles
    a_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
    b_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
    c_ty = np.ndarray[(m, n), np.dtype[dtype_out]]

    # AIE Core Function declarations
    func_type = "" if vectorized else "scalar_"
    bin_name = f"mm_{m}x{k}x{n}_{dtype_in_str}_{dtype_out_str}.o"
    zero_kernel = Kernel(
        f"zero_{func_type}{dtype_out_str}", bin_name, [c_ty]
    )
    matmul_vectorized_func_name = f"matmul_{dtype_in_str}_{dtype_out_str}"
    matmul_kernel = Kernel(
        matmul_vectorized_func_name,
        bin_name,
        [a_ty, b_ty, c_ty],
    )

    # AIE-array data movement with object fifos
    # Input A
    inA = ObjectFifo(a_ty, name="inA")
    a_dims = None
    if vectorized:
        a_dims = [(m // r, r * k), (k // s, s), (r, k), (s, 1)]
    memA = inA.cons().forward(name="memA", dims_to_stream=a_dims)

    # Input B
    inB = ObjectFifo(b_ty, name="inB")
    b_dims = None
    if vectorized:
        if transposed_b:
            b_dims = [(n // t, t * k), (k // s, s), (t, k), (s, 1)]
        else:
            b_dims = [(k // s, s * n), (n // t, t), (s, n), (t, 1)]
    memB = inB.cons().forward(name="memB", dims_to_stream=b_dims)

    # Output C
    memC = ObjectFifo(c_ty, name="memC")
    c_dims = None
    if vectorized:
        c_dims = [(m // r, r * n), (r, t), (n // t, r * t), (t, 1)]
    outC = memC.cons().forward(name="outC", dims_to_stream=c_dims)

    # Task each core will run
    def core_fn(of_a, of_b, of_c, zero, matmul):
        for _ in range_(tiles) if tiles > 1 else range(1):  # issue #1547
            elem_out = of_c.acquire(1)
            zero(elem_out)

            # issue #1547
            for _ in range_(K_div_k) if K_div_k > 1 else range(1):
                elem_in_a = of_a.acquire(1)
                elem_in_b = of_b.acquire(1)
                matmul(elem_in_a, elem_in_b, elem_out)
                of_a.release(1)
                of_b.release(1)
                
            of_c.release(1)

    # Create worker from task
    worker = Worker(
        core_fn, [memA.cons(), memB.cons(), memC.prod(), zero_kernel, matmul_kernel], stack_size=0xD00
    )
    
    # Define tensor access patterns for inputs/outputs
    # A and B are tiled across M and N respectively, while C is tiled across M and N
    A_tiles = TensorTiler2D.group_tiler(
        (heads * M, K), (m, k), (1, K_div_k)
    )
    
    if transposed_b:
        B_tiles = TensorTiler2D.group_tiler(
            (heads* N, K), (n, k), (1, K_div_k)
        )
    else:
        B_tiles = TensorTiler2D.group_tiler(
            (heads * K, N), (k, n), (K_div_k, 1), tile_group_col_major=True
        )

    C_tiles = TensorTiler2D.group_tiler((heads * M, N), (m, n), (1, 1))
        
    def print_tap_seq_info(tap_seq, name):
        for idx, tap in enumerate(tap_seq):
            print(f"{name} tile {idx}:")
            print(f"  Offset: {tap.offset}")
            print(f"  Sizes: {tap.sizes}")
            print(f"  Strides: {tap.strides}")

    if verbose:
        print(f"DMA Transfer Configuration: DRAM <-> Mem tile")
        print_tap_seq_info(A_tiles, "A")
        print_tap_seq_info(B_tiles, "B")
        print_tap_seq_info(C_tiles, "C")

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(A_ty, B_ty, C_ty) as (A, B, C):
        rt.enable_trace(trace_size, workers=[worker])
        rt.start(worker)

        A_idx = [i for i in range(heads * M_div_m) for _ in range(N_div_n)]  
        B_idx = [i + N_div_n*h for h in range(heads) for _ in range(M_div_m) for i in range(N_div_n)]# * N_div_n * heads

        for idx in range(len(C_tiles)):
            
            rt.fill(inA.prod(), A, tap=A_tiles[A_idx[idx]])
            rt.fill(inB.prod(), B, tap=B_tiles[B_idx[idx]])
            rt.drain(outC.cons(), C, tap=C_tiles[idx], wait=True)

    # Create the program from the device type and runtime
    if dev == "npu":
        dev_ty = NPU1Col1()
    else:
        dev_ty = NPU2()
    my_program = Program(dev_ty, rt)

    # Place components (assign them resources on the device) and generate an MLIR module
    module = my_program.resolve_program(SequentialPlacer())
    return module

if __name__ == "__main__":
    main()
