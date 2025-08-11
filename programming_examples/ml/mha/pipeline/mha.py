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
from aie.iron.device import NPU1Col1, NPU2, Tile
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorTiler2D

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
    s_ty = np.ndarray[(m,), np.dtype[dtype]]
    
    # AIE Core Function declarations
    func_type = "" if vectorized else "scalar_"
    bin_name = "kernels.a"
    
    
    zero_kernel = Kernel(
        f"zero_{func_type}{dtype_str}", bin_name, [qk_ty]
    )
    
    partial_softmax_kernel = Kernel(
        "partial_softmax",
        bin_name,
        [qk_ty, qk_ty, s_ty, np.float32, np.int32, np.int32],
    )
    
    matmul_QK = Kernel(
        "matmul_bf16_bf16",
        bin_name,
        [q_ty, k_ty, qk_ty],
    )
    
    matmul_PV = Kernel(
        "matmul_PV",
        bin_name,
        [qk_ty, k_ty, qk_ty, s_ty],
    )

    # AIE-array data movement with object fifos
    # Input Q
    inQ = ObjectFifo(q_ty, name="inQ")
    q_dims = None
    if vectorized:
        q_dims = [(m // r, r * k), (k // s, s), (r, k), (s, 1)]
    memQ = inQ.cons().forward(name="memQ", dims_to_stream=q_dims) # Forward DRAM -> Mem tile -> L1

    # Input K, in col major format (so we can skip the transpose)
    inK = ObjectFifo(k_ty, name="inK")
    k_dims = None
    if vectorized:
        k_dims = [(n // t, t * k), (k // s, s), (t, k), (s, 1)]
    memK = inK.cons().forward(name="memK", dims_to_stream=k_dims)
        
    # Input V
    inV = ObjectFifo(k_ty, name="inV")
    v_dims = None
    if vectorized:
        v_dims = [(k // s, s * n), (n // t, t), (s, n), (t, 1)]
    memV = inV.cons().forward(name="memV", dims_to_stream=v_dims, placement=Tile(col=1, row=1))

    # Output QK
    qk_dims = None
    if vectorized:
        qk_dims = [(m // r, r * n), (r, t), (n // t, r * t), (t, 1)]
    memQK = ObjectFifo(qk_ty, name="memQK")
    outQK = memQK.cons().forward(name="outQK", dims_to_stream=qk_dims)
    
    # Output A
    memA = ObjectFifo(qk_ty, name="memA")
    outA = memA.cons().forward(name="outA", dims_to_stream=q_dims, placement=Tile(col=1, row=1))
    
    # Scale buffer for partial softmax
    scaleOF = ObjectFifo(s_ty, name="scaleOF")
    
    # Output O
    memO = ObjectFifo(qk_ty, name="memO")
    o_dims = None
    if vectorized:
        o_dims = [(m // r, r * n), (r, t), (n // t, r * t), (t, 1)]
    outO = memO.cons().forward(name="outO", dims_to_stream=o_dims, placement=Tile(col=1, row=1))
    
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

    def batched_matmul_qk(of_q, of_k, of_qk_out, zero, matmul_QK):
        
        for _ in range_(tiles):
            elem_qk_out = of_qk_out.acquire(1)
            zero(elem_qk_out)

            elem_in_q = of_q.acquire(1)
            elem_in_k = of_k.acquire(1)
            
            matmul_QK(elem_in_q, elem_in_k, elem_qk_out)
            
            of_q.release(1)
            of_k.release(1)
                
            of_qk_out.release(1)

    def partial_softmax(of_in_a, of_out_b, of_out_scale, softmax):
        
        elt_of_out_b = of_out_b.acquire(1)
        elt_of_in_a = of_in_a.acquire(1)
        elt_of_out_scale = of_out_scale.acquire(1)
        
        softmax(elt_of_in_a, elt_of_out_b, elt_of_out_scale, inv_scale, S_q, S_kv)
        
        of_in_a.release(1)
        of_out_b.release(1)
        of_out_scale.release(1)
    
    def batched_matmul_av(of_a, of_v, of_scale, of_o_out, zero, matmul_PV):
        
        
        elt_of_out_scale = of_scale.acquire(1)
        
        for _ in range_(tiles):
            elem_o_out = of_o_out.acquire(1)
            zero(elem_o_out)

            elem_in_a = of_a.acquire(1)
            elem_in_v = of_v.acquire(1)
            
            matmul_PV(elem_in_a, elem_in_v, elem_o_out, elt_of_out_scale)
            
            of_a.release(1)
            of_v.release(1)
                
            of_o_out.release(1)
        
        of_scale.release(1)

    # Create worker from task
    matmul_worker = Worker(
        batched_matmul_qk,
        fn_args = [
            memQ.cons(), 
            memK.cons(),
            memQK.prod(),
            zero_kernel,
            matmul_QK,
        ], 
        stack_size=0xD00,
        placement=Tile(col=0, row=2)
    )
    
    softmax_worker = Worker(
        partial_softmax,
        fn_args = [
            outQK.cons(),
            memA.prod(),
            scaleOF.prod(),
            partial_softmax_kernel,
        ],
        stack_size=0xD00,
        placement=Tile(col=0, row=3)
    )
    
    matmul_av_worker = Worker(
        batched_matmul_av,
        fn_args = [
            outA.cons(),
            memV.cons(),
            scaleOF.cons(),
            memO.prod(),
            zero_kernel,
            matmul_PV,
        ], 
        stack_size=0xD00,
        placement=Tile(col=0, row=4)
    )
    
    # Define tensor access patterns for inputs/outputs
    # A and B are tiled across M and N respectively, while C is tiled across M and N
    Q_tiles = TensorTiler2D.group_tiler((heads * S_q, d), (m, k), (1, d_div_k))
    
    K_tiles = TensorTiler2D.group_tiler((heads* S_kv, d), (n, k), (1, d_div_k))
    
    V_tiles = TensorTiler2D.group_tiler(
        (heads * d, S_kv), (k, n), (d_div_k, 1), 
        tile_group_col_major=True
    )

    QK_tiles = TensorTiler2D.group_tiler((heads * S_q, S_kv), (m, n), (1, 1))
    
    O_tiles = TensorTiler2D.group_tiler((heads * S_q, d), (m, n), (1, 1))
        
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
        print_tap_seq_info(V_tiles, "V")
        print_tap_seq_info(QK_tiles, "C")

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(Q_ty, KV_ty, KV_ty, KV_ty) as (Q, K, V, O):
        rt.start(matmul_worker)
        rt.start(softmax_worker)
        rt.start(matmul_av_worker)

        Q_idx = [i for i in range(heads * S_q_div_m) for _ in range(S_kv_div_n)]  
        K_idx = [i + S_kv_div_n*h for h in range(heads) for _ in range(S_q_div_m) for i in range(S_kv_div_n)]

        for idx in range(len(QK_tiles)):
            
            rt.fill(inQ.prod(), Q, tap=Q_tiles[Q_idx[idx]], placement = Tile(col = 0, row = 0))
            rt.fill(inK.prod(), K, tap=K_tiles[K_idx[idx]], placement = Tile(col = 0, row = 0))
            rt.fill(inV.prod(), V, tap=V_tiles[K_idx[idx]], placement = Tile(col = 1, row = 0))
            rt.drain(outO.cons(), O, tap=O_tiles[idx], wait=True, placement = Tile(col = 0, row = 0))
            

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
