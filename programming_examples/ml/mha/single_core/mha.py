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

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker, GlobalBuffer, WorkerRuntimeBarrier
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col1, NPU2
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorTiler2D

from aie.helpers.dialects.ext.scf import if_, else_

base_dir = Path(__file__).parent

dtype_map = {
    "bf16": bfloat16,
    "f32": np.float32,
}

microkernel_mac_dim_map = {
    "npu": {
        "bf16": (4, 8, 4),
    },
    "npu2": {
        "bf16": {
            # emulate_bf16_mmul_with_bfp16
            True: (8, 8, 8),
            False: (4, 8, 8),
        },
    },
}

def main():
    argparser = argparse.ArgumentParser(
        prog="AIE Matrix Multiplication MLIR Design (Single Core)",
        description="Emits MLIR code for a matrix multiplication design of the given input size",
    )
    argparser.add_argument("--dev", type=str, choices=["npu", "npu2"], default="npu")
    argparser.add_argument("--heads", type=int, default=1)
    argparser.add_argument("--S_q", type=int, default=256)
    argparser.add_argument("--S_kv", type=int, default=256)
    argparser.add_argument("-d", type=int, default=64)
    argparser.add_argument("-m", type=int, default=64)
    argparser.add_argument("-k", type=int, default=64)
    argparser.add_argument("-n", type=int, default=64)
    argparser.add_argument(
        "--dtype", type=str, choices=["bf16", "f32"], default="bf16"
    )
    argparser.add_argument("--emulate-bf16-mmul-with-bfp16", type=bool, default=False)
    argparser.add_argument("--trace_size", type=int, default=0)
    argparser.add_argument("--output_file_path", type=str, default = base_dir / "build" / f"my_mha.mlir", help="Output file path for the generated MLIR module")
    argparser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = argparser.parse_args()
    maybe_module = batched_matmul_single_core(
        args.dev,
        args.heads,
        args.S_q,
        args.S_kv,
        args.d,
        args.m,
        args.k,
        args.n,
        args.dtype,
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
    S_q: int,
    S_kv: int,
    d: int,
    m: int,
    k: int,
    n: int,
    dtype_str: str,
    emulate_bf16_mmul_with_bfp16: bool,
    trace_size: int = 0,
    verbose: bool = False,
):

    vectorized = True
    enable_tracing = True if trace_size > 0 else False

    # r, s, t are the dimensions required by the microkernel MAC instructions.
    mac_dims = microkernel_mac_dim_map[dev][dtype_str]
    if dev == "npu2" and dtype_str == "bf16":
        r, s, t = mac_dims[emulate_bf16_mmul_with_bfp16]
    else:
        r, s, t = mac_dims

    if verbose:
        print(f"Device: {dev}")
        print(f"Number of heads: {heads}")
        print(f"MHA Dimensions: S_q={S_q}, S_kv={S_kv}, d={d}, m={m}, k={k}, n={n}")
        print(f"Data type: {dtype_str}")
        print(f"Microkernel MAC dimensions: r={r}, s={s}, t={t}")
        print(f"Vectorized: {vectorized}")
        print(f"Enable tracing: {enable_tracing}")
        
    assert heads > 0, "Number of heads must be greater than 0"
    assert S_q % m == 0, f"M must be divisible by m ({S_q} % {m} != 0)"
    assert S_kv % n == 0, f"N must be divisible by n ({S_kv} % {n} != 0)"
    assert d % k == 0, f"K must be divisible by k ({d} % {k} != 0)"
    
    assert m % r == 0, f"m must be divisible by r ({m} % {r} != 0)"
    assert k % s == 0, f"k must be divisible by s ({k} % {s} != 0)"
    assert n % t == 0, f"n must be divisible by t ({n} % {t} != 0)"

    dtype = dtype_map[dtype_str]

    assert np.issubdtype(dtype, np.integer) == np.issubdtype(
        dtype, np.integer
    ), f"Input dtype ({dtype}) and output dtype ({dtype}) must either both be integer or both be float"
    assert (
        np.dtype(dtype).itemsize >= np.dtype(dtype).itemsize
    ), f"Output dtype ({dtype}) must be equal or larger to input dtype ({dtype})"

    S_q_div_m = S_q // m
    S_kv_div_n = S_kv // n
    d_div_k = d // k
    tiles = heads * S_q_div_m * S_kv_div_n
    
    inv_scale = 1 / np.sqrt(d)

    # Tensors living in DRAM
    Q_ty = np.ndarray[(heads * S_q * d,), np.dtype[dtype]]
    KV_ty = np.ndarray[(heads * S_kv * d,), np.dtype[dtype]]
    A_ty = np.ndarray[(heads * S_q * S_kv,), np.dtype[dtype]]
    
    # Tensors living in Mem Tiles
    q_ty = np.ndarray[(m, k), np.dtype[dtype]]
    k_ty = np.ndarray[(k, n), np.dtype[dtype]]
    qk_ty = np.ndarray[(m, n), np.dtype[dtype]]

    # AIE Core Function declarations
    func_type = "" if vectorized else "scalar_"
    # bin_name = f"mm_{m}x{k}x{n}_{dtype_str}_{dtype_str}.o"
    bin_name = "kernels.a"
    zero_kernel = Kernel(
        f"zero_{func_type}{dtype_str}", bin_name, [qk_ty]
    )
    # matmul_vectorized_func_name = f"matmul_{dtype_str}_{dtype_str}"
    matmul_vectorized_func_name = "mha_bf16_bf16"
    print(f"Using matmul function: {matmul_vectorized_func_name}")
    matmul_kernel = Kernel(
        matmul_vectorized_func_name,
        bin_name,
        [q_ty, k_ty, qk_ty, np.int32, np.float32, np.int32, np.int32],
    )

    # AIE-array data movement with object fifos
    # Input Q
    inQ = ObjectFifo(q_ty, name="inQ")
    q_dims = None
    if vectorized:
        q_dims = [(m // r, r * k), (k // s, s), (r, k), (s, 1)]
    memQ = inQ.cons().forward(name="memQ", dims_to_stream=q_dims) # Forward DRAM -> Mem tile -> L1

    # Input K
    inK = ObjectFifo(k_ty, name="inK")
    k_dims = None
    if vectorized:
        k_dims = [(n // t, t * k), (k // s, s), (t, k), (s, 1)]

    memK = inK.cons().forward(name="memK", dims_to_stream=k_dims)

    # Output QK
    memQK = ObjectFifo(qk_ty, name="memQK")
    qk_dims = None
    if vectorized:
        qk_dims = [(m // r, r * n), (r, t), (n // t, r * t), (t, 1)]
    outQK = memQK.cons().forward(name="outQK", dims_to_stream=qk_dims)
    
    # Runtime parameters to control which microkernel to toggle
    rtps = []
    rtps.append(
        GlobalBuffer(
            np.ndarray[(1,), np.dtype[np.int32]],
            name="rtp",
            initial_value=np.array([1], dtype=np.int32),
            use_write_rtp=True,
        )
    )
    
    workerBarriers = []
    workerBarriers.append(WorkerRuntimeBarrier())

    # Task each core will run
    def core_fn(of_q, of_k, of_qk_out, zero, mha, rtp, barrier):
        
        barrier.wait_for_value(1)
        
        with if_(rtp[0] == 1) as if_op:
            for _ in range_(tiles) if tiles > 1 else range(1):  # issue #1547
                elem_qk_out = of_qk_out.acquire(1)
                zero(elem_qk_out)

                # issue #1547
                for _ in range_(d_div_k) if d_div_k > 1 else range(1):
                    elem_in_q = of_q.acquire(1)
                    elem_in_k = of_k.acquire(1)
                    mha(elem_in_q, elem_in_k, elem_qk_out, rtp[0], inv_scale, S_q, S_kv)
                    of_q.release(1)
                    of_k.release(1)
                
            of_qk_out.release(1)
        with else_(if_op):
            # Reconfigure of configuration here
            # Write directly into the config register -> What address should I write to? 
            # write32(Address addr, uint32_t value)
            elem_out_a = of_q.acquire(1)
            elem_in_qk = of_qk_out.acquire(1)
            mha(elem_in_qk, elem_in_qk, elem_out_a, rtp[0], inv_scale, S_q, S_kv)
            of_q.release(1)
            of_qk_out.release(1)
            
        barrier.release_with_value(1)
    

    # Create worker from task
    worker = Worker(
        core_fn, 
        fn_args = [
            memQ.cons(), 
            memK.cons(),
            memQK.prod(),
            zero_kernel,
            matmul_kernel,
            rtps[0],
            workerBarriers[0],
        ], 
        stack_size=0xD00
    )
    
    # Define tensor access patterns for inputs/outputs
    # A and B are tiled across M and N respectively, while C is tiled across M and N
    Q_tiles = TensorTiler2D.group_tiler(
        (heads * S_q, d), (m, k), (1, d_div_k)
    )
    
    K_tiles = TensorTiler2D.group_tiler(
        (heads* S_kv, d), (n, k), (1, d_div_k)
    )

    QK_tiles = TensorTiler2D.group_tiler((heads * S_q, S_kv), (m, n), (1, 1))
        
    def print_tap_seq_info(tap_seq, name):
        for idx, tap in enumerate(tap_seq):
            print(f"{name} tile {idx}:")
            print(f"  Offset: {tap.offset}")
            print(f"  Sizes: {tap.sizes}")
            print(f"  Strides: {tap.strides}")

    if verbose:
        print(f"DMA Transfer Configuration: DRAM <-> Mem tile")
        print_tap_seq_info(Q_tiles, "A")
        print_tap_seq_info(K_tiles, "B")
        print_tap_seq_info(QK_tiles, "C")

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(Q_ty, KV_ty, A_ty, A_ty) as (Q, K, QK, A):
        # rt.enable_trace(trace_size, workers=[worker])
        rt.start(worker)
        
        # 1 -> Batched MatMul colmajor
        def set_rtps(*args):
            for rtp in args:
                rtp[0] = 1
                
        rt.inline_ops(set_rtps, rtps)
        rt.set_barrier(workerBarriers[0], 1)

        Q_idx = [i for i in range(heads * S_q_div_m) for _ in range(S_kv_div_n)]  
        K_idx = [i + S_kv_div_n*h for h in range(heads) for _ in range(S_q_div_m) for i in range(S_kv_div_n)]

        for idx in range(len(QK_tiles)):
            
            rt.fill(inQ.prod(), Q, tap=Q_tiles[Q_idx[idx]])
            rt.fill(inK.prod(), K, tap=K_tiles[K_idx[idx]])
            rt.drain(outQK.cons(), QK, tap=QK_tiles[idx], wait=True)
            
        # 2 -> Softmax
        def set_rtps(*args):
            for rtp in args:
                rtp[0] = 2
                
        rt.inline_ops(set_rtps, rtps)
        rt.set_barrier(workerBarriers[0], 1)
        
        rt.fill(inQ.prod(), QK)
        rt.drain(outQK.cons(), A, wait=True)

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
