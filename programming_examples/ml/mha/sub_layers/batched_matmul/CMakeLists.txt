# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

cmake_minimum_required(VERSION 3.20)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# Configuration variables
set(heads 8)
set(M 512)
set(K 64)
set(N 512)
set(m 32)
set(k 32)
set(n 32)
set(dtype_in "bf16")
set(dtype_out "bf16")
set(transposed_b 1)

set(verbose 0)
set(trace_size 65536)
set(device_name "npu2")
set(design_name "batched_matmul_single_core")

set(ProjectName ${design_name})
set(currentDesignTarget ${design_name})

project(${ProjectName})

# Derived variables
set(target_suffix "${heads}_${M}x${K}x${N}_${m}x${k}x${n}")
set(kernel "mm_${m}x${k}x${n}_${dtype_in}_${dtype_out}")

if(verbose STREQUAL "1")
    set(verbose_string "--verbose")
else()
    set(verbose_string "")
endif()

# Data type mappings
if(dtype_in STREQUAL "i16")
    set(dtype_in_cpp "int16_t")
elseif(dtype_in STREQUAL "i8")
    set(dtype_in_cpp "int8_t")
elseif(dtype_in STREQUAL "bf16")
    set(dtype_in_cpp "std::bfloat16_t")
elseif(dtype_in STREQUAL "f32")
    set(dtype_in_cpp "float")
endif()

if(dtype_out STREQUAL "i32")
    set(dtype_out_cpp "int32_t")
    set(dtype_acc_cpp "int32_t")
elseif(dtype_out STREQUAL "i16")
    set(dtype_out_cpp "int16_t")
    set(dtype_acc_cpp "int16_t")
elseif(dtype_out STREQUAL "bf16")
    set(dtype_out_cpp "std::bfloat16_t")
    set(dtype_acc_cpp "float")
elseif(dtype_out STREQUAL "f32")
    set(dtype_out_cpp "float")
    set(dtype_acc_cpp "float")
elseif(dtype_out STREQUAL "i8")
    set(dtype_out_cpp "int8_t")
    set(dtype_acc_cpp "int8_t")
endif()

# Display configuration
message(STATUS "Design Configuration:")
message(STATUS "  Heads: ${heads}")
message(STATUS "  MatMul dimensions: ${M}x${K}x${N}")
message(STATUS "  Tile dimensions: ${m}x${k}x${n}")
message(STATUS "  Data types: ${dtype_in} -> ${dtype_out}")
message(STATUS "  Device: ${device_name}")
message(STATUS "  Design: ${design_name}")
message(STATUS "  Verbose mode: ${verbose}")

# Platform detection
find_program(WSL NAMES powershell.exe)

if (NOT WSL)
    set(CMAKE_C_COMPILER gcc-13)
    set(CMAKE_CXX_COMPILER g++-13)
    set(XRT_INC_DIR /opt/xilinx/xrt/include CACHE STRING "Path to XRT include")
    set(XRT_LIB_DIR /opt/xilinx/xrt/lib CACHE STRING "Path to XRT libraries")
else()
    set(XRT_INC_DIR C:/Technical/XRT/src/runtime_src/core/include CACHE STRING "Path to XRT include")
    set(XRT_LIB_DIR C:/Technical/xrtNPUfromDLL CACHE STRING "Path to XRT libraries")
endif()

# Environment validation
if(NOT DEFINED ENV{PEANO_INSTALL_DIR})
    message(FATAL_ERROR "PEANO_INSTALL_DIR environment variable not set. Please source utils/env_setup.sh")
endif()

# Find required tools
find_program(python_executable python REQUIRED)
find_program(aiecc_executable aiecc.py REQUIRED)

# Path configuration
set(kernels_dir "${CMAKE_SOURCE_DIR}/../../../../../aie_kernels/aie2p")
set(runtime_lib_dir "${CMAKE_SOURCE_DIR}/../../../../../runtime_lib")
set(test_lib_dir "${runtime_lib_dir}/test_lib")

find_program(aieopt_dir aie-opt)

if(aieopt_dir)
    # Get the parent directory of the binary
    get_filename_component(mlir_aie_bin_dir ${aieopt_dir} DIRECTORY)
    # Get the parent directory of the parent directory
    get_filename_component(mlir_aie_dir ${mlir_aie_bin_dir} DIRECTORY)
else()
    message(FATAL_ERROR "aie-opt not found in PATH")
endif()

# Build artifact paths
set(golden_ref_header "${CMAKE_BINARY_DIR}/golden_reference.h")
set(mlir_target "${CMAKE_BINARY_DIR}/aie_${target_suffix}.mlir")
set(trace_mlir_target "${CMAKE_BINARY_DIR}/aie_trace_${target_suffix}.mlir")
set(xclbin_target "${CMAKE_BINARY_DIR}/final_${target_suffix}.xclbin")
set(trace_xclbin_target "${CMAKE_BINARY_DIR}/trace_${target_suffix}.xclbin")
set(insts_target "${CMAKE_BINARY_DIR}/insts_${target_suffix}.txt")
set(kernel_obj "${CMAKE_BINARY_DIR}/${kernel}.o")

# Kernel compilation flags
set(kernel_cc "$ENV{PEANO_INSTALL_DIR}/bin/clang++")
set(warning_flags 
    "-Wno-parentheses"
    "-Wno-attributes"
    "-Wno-macro-redefined"
    "-Wno-empty-body"
    "-Wno-unknown-attributes")
set(kernel_defines
    "-D${dtype_in}_${dtype_out}_ONLY"
    "-DDIM_M=${m}" 
    "-DDIM_K=${k}" 
    "-DDIM_N=${n}"
)

if(transposed_b STREQUAL "1")
    list(APPEND kernel_defines "-DB_COL_MAJ")
endif()

if(device_name STREQUAL "npu")
    set(kernel_cflags_base 
        "-O2" "-std=c++20" "--target=aie2-none-unknown-elf"
        ${warning_flags} "-DNDEBUG" "-I${mlir_aie_dir}/include"
    )
elseif(device_name STREQUAL "npu2")
    set(kernel_cflags_base 
        "-O2" "-std=c++20" "--target=aie2p-none-unknown-elf"
        ${warning_flags} "-DNDEBUG" "-I${mlir_aie_dir}/include"
    )
endif()

set(kernel_cflags ${kernel_cflags_base} ${kernel_defines})

# AIE compiler flags
set(aiecc_flags
    "--alloc-scheme=basic-sequential"
    "--aie-generate-xclbin"
    "--no-compile-host"
    "--no-xchesscc"
    "--no-xbridge"
    "--peano" "$ENV{PEANO_INSTALL_DIR}"
    "--aie-generate-npu-insts"
)

# Always use PyTorch golden reference
add_custom_command(
    OUTPUT ${golden_ref_header}
    COMMAND ${python_executable} ${CMAKE_SOURCE_DIR}/generate_golden_reference.py
        --heads ${heads} -M ${M} -K ${K} -N ${N}
        --dtype_in ${dtype_in} --dtype_out ${dtype_out}
        --transposed_b ${transposed_b}
        --output ${golden_ref_header} ${verbose_string}
    DEPENDS ${CMAKE_SOURCE_DIR}/generate_golden_reference.py
    COMMENT "Generating PyTorch golden reference"
)

# Kernel compilation
add_custom_command(
    OUTPUT ${kernel_obj}
    COMMAND ${kernel_cc} ${kernel_cflags} 
        -c ${kernels_dir}/mm.cc -o ${kernel_obj}
    DEPENDS ${kernels_dir}/mm.cc
    COMMENT "Compiling AIE kernel: ${kernel}"
)

# MLIR generation
add_custom_command(
    OUTPUT ${mlir_target}
    COMMAND ${python_executable} ${CMAKE_SOURCE_DIR}/${design_name}.py
        --heads ${heads}
        -m ${m} -k ${k} -n ${n}
        --dev ${device_name} -M ${M} -K ${K} -N ${N}
        --dtype_in ${dtype_in} --dtype_out ${dtype_out}
        --transposed_b ${transposed_b}
        --trace_size 0 ${verbose_string} --output_file_path ${mlir_target}
    DEPENDS ${CMAKE_SOURCE_DIR}/${design_name}.py
    COMMENT "Generating MLIR for ${target_suffix}"
)

# Trace MLIR generation
add_custom_command(
    OUTPUT ${trace_mlir_target}
    COMMAND ${python_executable} ${CMAKE_SOURCE_DIR}/${design_name}.py
        --heads ${heads} -m ${m} -k ${k} -n ${n}
        --dev ${device_name} -M ${M} -K ${K} -N ${N}
        --dtype_in ${dtype_in} --dtype_out ${dtype_out}
        --trace_size ${trace_size} ${verbose_string} --output_file_path ${trace_mlir_target}
    DEPENDS ${CMAKE_SOURCE_DIR}/${design_name}.py
    COMMENT "Generating trace MLIR for ${target_suffix}"
)

# XCLBIN generation
add_custom_command(
    OUTPUT ${xclbin_target} ${insts_target}
    COMMAND ${aiecc_executable} ${aiecc_flags}
        --xclbin-name=final_${target_suffix}.xclbin
        --npu-insts-name=insts_${target_suffix}.txt
        ${mlir_target}
    DEPENDS ${mlir_target} ${kernel_obj}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Generating XCLBIN and instructions for ${target_suffix}"
)

# Trace XCLBIN generation
add_custom_command(
    OUTPUT ${trace_xclbin_target}
    COMMAND ${aiecc_executable} ${aiecc_flags}
        --xclbin-name=trace_${target_suffix}.xclbin
        --npu-insts-name=insts_${target_suffix}.txt
        ${trace_mlir_target}
    DEPENDS ${trace_mlir_target} ${kernel_obj}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Generating trace XCLBIN for ${target_suffix}"
)

# Host executable
add_executable(${currentDesignTarget}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../runtime_lib/test_lib/test_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test.cpp
)

target_compile_definitions(${currentDesignTarget} PUBLIC 
    DISABLE_ABI_CHECK=1
    DTYPE_IN=${dtype_in_cpp}
    DTYPE_OUT=${dtype_out_cpp}
    DTYPE_ACC=${dtype_acc_cpp}
)

target_include_directories (${currentDesignTarget} PUBLIC
    ${XRT_INC_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../runtime_lib/test_lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../runtime_lib
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_BINARY_DIR}
)

target_link_directories(${currentDesignTarget} PUBLIC
    ${XRT_LIB_DIR}
)

target_link_libraries(${currentDesignTarget} PUBLIC
    xrt_coreutil
)

target_compile_options(${currentDesignTarget} PRIVATE
    -ggdb
)

# Ensure golden reference is generated before compiling
add_dependencies(${currentDesignTarget} golden_ref)

# Custom targets
add_custom_target(golden_ref DEPENDS ${golden_ref_header})
add_custom_target(xclbin DEPENDS ${xclbin_target} ${insts_target})
add_custom_target(trace_xclbin DEPENDS ${trace_xclbin_target})

# Run target
add_custom_target(run
    COMMAND ${CMAKE_COMMAND} -E env XRT_HACK_UNSECURE_LOADING_XCLBIN=1
        $<TARGET_FILE:${currentDesignTarget}>
        -x ${xclbin_target}
        -i ${insts_target}
        -k MLIR_AIE
        --heads ${heads}
        -M ${M} -K ${K} -N ${N}
        -v 2 --warmup 1 --iters 1
        --verbosity ${verbose}
    DEPENDS ${currentDesignTarget} xclbin
    COMMENT "Running ${currentDesignTarget} test"
)

# Trace target
add_custom_target(trace
    COMMAND ${CMAKE_COMMAND} -E env XRT_HACK_UNSECURE_LOADING_XCLBIN=1
        $<TARGET_FILE:${currentDesignTarget}>
        -x ${trace_xclbin_target}
        -i ${insts_target}
        -k MLIR_AIE
        --heads ${heads}
        -M ${M} -K ${K} -N ${N}
        -v 2 --warmup 1 --iters 1
        -t ${trace_size}
        --verbosity ${verbose}
    DEPENDS ${currentDesignTarget} trace_xclbin
    COMMENT "Running ${currentDesignTarget} test with tracing"
)

# Parse trace target
add_custom_target(parse_trace
    COMMAND ${python_executable} ${CMAKE_SOURCE_DIR}/../../../utils/parse_trace.py
        --input trace.txt
        --mlir ${trace_mlir_target}
        --output trace_mm.json
    COMMENT "Parsing trace output"
)

# Clean targets
add_custom_target(clean_all
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}
    COMMENT "Cleaning all build artifacts"
)
