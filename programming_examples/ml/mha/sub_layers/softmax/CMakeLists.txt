# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

cmake_minimum_required(VERSION 3.20)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# Configuration variables
set(N 262144 CACHE STRING "Vector size")
set(n 1024 CACHE STRING "Tile size")
set(n_cores 2 CACHE STRING "Number of cores")
set(dtype_in "bf16" CACHE STRING "Input data type")
set(dtype_out "bf16" CACHE STRING "Output data type")
set(trace_size 8192 CACHE STRING "Trace buffer size")
set(verbose 0 CACHE STRING "Verbosity level")
set(devicename "npu2" CACHE STRING "Device name")
set(TARGET_NAME "softmax" CACHE STRING "Target name to be built")

# Derived variables
set(target_suffix "${N}_${n}_${n_cores}")
set(kernels "softmax")

# Data type mappings
if(dtype_in STREQUAL "bf16")
    set(dtype_in_cpp "std::bfloat16_t")
elseif(dtype_in STREQUAL "f32")
    set(dtype_in_cpp "float")
endif()

if(dtype_out STREQUAL "bf16")
    set(dtype_out_cpp "std::bfloat16_t")
elseif(dtype_out STREQUAL "f32")
    set(dtype_out_cpp "float")
endif()

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

SET (ProjectName ${TARGET_NAME})
SET (currentTarget ${TARGET_NAME})

if ( WSL )
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR})
    add_compile_options(/Zc:__cplusplus)
endif ()

project(${ProjectName})

# Environment validation
if(NOT DEFINED ENV{PEANO_INSTALL_DIR})
    message(FATAL_ERROR "PEANO_INSTALL_DIR environment variable not set. Please source utils/env_setup.sh")
endif()

# Find required tools
find_program(PYTHON_EXECUTABLE python3 REQUIRED)
find_program(AIECC_EXECUTABLE aiecc.py REQUIRED)

find_program(aieopt_dir aie-opt)

if(aieopt_dir)
    # Get the parent directory of the binary
    get_filename_component(mlir_aie_bin_dir ${aieopt_dir} DIRECTORY)
    # Get the parent directory of the parent directory
    get_filename_component(mlir_aie_dir ${mlir_aie_bin_dir} DIRECTORY)
else()
    message(FATAL_ERROR "aie-opt not found in PATH")
endif()

# Path configuration
if(devicename STREQUAL "npu")
    set(kernels_dir "${CMAKE_SOURCE_DIR}/../../../../../aie_kernels/aie2")
    set(aie2_runtime_dir "$ENV{AIEOPT_DIR}/aie_runtime_lib/AIE2")
elseif(devicename STREQUAL "npu2")
    set(kernels_dir "${CMAKE_SOURCE_DIR}/../../../../../aie_kernels/aie2p")
else()
    message(FATAL_ERROR "Device type ${devicename} not supported")
endif()

set(runtime_lib_dir "${CMAKE_SOURCE_DIR}/../../../../../runtime_lib")
set(test_lib_dir "${runtime_lib_dir}/test_lib")

# Build artifact paths
set(mlir_target "${CMAKE_BINARY_DIR}/aie_${target_suffix}.mlir")
set(trace_mlir_target "${CMAKE_BINARY_DIR}/aie_trace_${target_suffix}.mlir")
set(xclbin_target "${CMAKE_BINARY_DIR}/final_${target_suffix}.xclbin")
set(trace_xclbin_target "${CMAKE_BINARY_DIR}/trace_${target_suffix}.xclbin")
set(insts_target "${CMAKE_BINARY_DIR}/insts_${target_suffix}.txt")
set(kernel_obj "${CMAKE_BINARY_DIR}/softmax.o")

# Kernel compilation flags
set(KERNEL_CC "$ENV{PEANO_INSTALL_DIR}/bin/clang++")
if(devicename STREQUAL "npu")
    set(KERNEL_CFLAGS 
        "-O2" "-std=c++20" "--target=aie2-none-unknown-elf"
        "-I$ENV{PEANO_INSTALL_DIR}/../include"
        "-I${aie2_runtime_dir}"
        "-I${mlir_aie_dir}/include"
        "-c"
    )
    set(lut_ops_obj "${CMAKE_BINARY_DIR}/lut_based_ops.o")
elseif(devicename STREQUAL "npu2")
    set(KERNEL_CFLAGS 
        "-O2" "-std=c++20" "--target=aie2p-none-unknown-elf"
        "-I$ENV{PEANO_INSTALL_DIR}/../include"
        "-I${mlir_aie_dir}/include"
        "-c"
    )
endif()

# AIE compiler flags
set(AIECC_FLAGS
    "--alloc-scheme=basic-sequential"
    "--aie-generate-xclbin"
    "--no-compile-host"
    "--no-xchesscc"
    "--no-xbridge"
    "--peano" "$ENV{PEANO_INSTALL_DIR}"
    "--aie-generate-npu-insts"
)

# Kernel compilation for NPU
if(devicename STREQUAL "npu")
    add_custom_command(
        OUTPUT ${lut_ops_obj}
        COMMAND ${KERNEL_CC} ${KERNEL_CFLAGS} 
            ${aie2_runtime_dir}/lut_based_ops.cpp -o ${lut_ops_obj}
        DEPENDS ${aie2_runtime_dir}/lut_based_ops.cpp
        COMMENT "Compiling lut_based_ops for NPU"
    )
endif()

add_custom_command(
    OUTPUT ${kernel_obj}
    COMMAND ${KERNEL_CC} ${KERNEL_CFLAGS} 
        ${kernels_dir}/softmax.cc -o ${kernel_obj}
    DEPENDS ${kernels_dir}/softmax.cc
    COMMENT "Compiling AIE kernel: ${kernels}"
)

# Create kernels.a archive
if(devicename STREQUAL "npu")
    add_custom_command(
        OUTPUT kernels.a
        COMMAND ar rvs kernels.a ${kernel_obj} ${lut_ops_obj}
        DEPENDS ${kernel_obj} ${lut_ops_obj}
        COMMENT "Creating kernels archive for NPU"
    )
elseif(devicename STREQUAL "npu2")
    add_custom_command(
        OUTPUT kernels.a
        COMMAND ar rvs kernels.a ${kernel_obj}
        DEPENDS ${kernel_obj}
        COMMENT "Creating kernels archive for NPU2"
    )
endif()

# MLIR generation
add_custom_command(
    OUTPUT ${mlir_target}
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/softmax.py
        ${devicename} 0 > ${mlir_target}
    DEPENDS ${CMAKE_SOURCE_DIR}/softmax.py
    COMMENT "Generating MLIR for ${target_suffix}"
)

# Trace MLIR generation
add_custom_command(
    OUTPUT ${trace_mlir_target}
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/softmax.py
        ${devicename} ${trace_size} > ${trace_mlir_target}
    DEPENDS ${CMAKE_SOURCE_DIR}/softmax.py
    COMMENT "Generating trace MLIR for ${target_suffix}"
)

# XCLBIN generation
add_custom_command(
    OUTPUT ${xclbin_target} ${insts_target}
    COMMAND ${AIECC_EXECUTABLE} ${AIECC_FLAGS}
        --xclbin-name=final_${target_suffix}.xclbin
        --npu-insts-name=insts_${target_suffix}.txt
        ${mlir_target}
    DEPENDS ${mlir_target} kernels.a
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Generating XCLBIN and instructions for ${target_suffix}"
)

# Trace XCLBIN generation
add_custom_command(
    OUTPUT ${trace_xclbin_target}
    COMMAND ${AIECC_EXECUTABLE} ${AIECC_FLAGS}
        --xclbin-name=trace_${target_suffix}.xclbin
        --npu-insts-name=insts_${target_suffix}.txt
        ${trace_mlir_target}
    DEPENDS ${trace_mlir_target} kernels.a
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Generating trace XCLBIN for ${target_suffix}"
)

add_custom_command(
    OUTPUT golden_reference.h
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate_golden_reference.py
    DEPENDS generate_golden_reference.py
    COMMENT "Generating PyTorch golden reference for softmax"
)

# Host executable
add_executable(${currentTarget}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../runtime_lib/test_lib/test_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test.cpp
)

add_custom_target(generate_golden_reference ALL DEPENDS golden_reference.h)
add_dependencies(softmax generate_golden_reference)

target_compile_definitions(${currentTarget} PUBLIC 
    DISABLE_ABI_CHECK=1
    DTYPE_IN=${dtype_in_cpp}
    DTYPE_OUT=${dtype_out_cpp}
)

target_include_directories (${currentTarget} PUBLIC
    ${XRT_INC_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../runtime_lib/test_lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../runtime_lib
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_BINARY_DIR}
)

target_link_directories(${currentTarget} PUBLIC
    ${XRT_LIB_DIR}
)

target_link_libraries(${currentTarget} PUBLIC
    xrt_coreutil
)

target_compile_options(${currentTarget} PRIVATE
    -ggdb
)

# Custom targets
add_custom_target(xclbin DEPENDS ${xclbin_target} ${insts_target})
add_custom_target(trace_xclbin DEPENDS ${trace_xclbin_target})

# Run target
add_custom_target(run
    COMMAND ${CMAKE_COMMAND} -E env XRT_HACK_UNSECURE_LOADING_XCLBIN=1
        $<TARGET_FILE:${currentTarget}>
        -x ${xclbin_target}
        -i ${insts_target}
        -k MLIR_AIE
        --verbosity ${verbose}
    DEPENDS ${currentTarget} xclbin
    COMMENT "Running ${currentTarget} test"
)

# Trace target
add_custom_target(trace
    COMMAND ${CMAKE_COMMAND} -E env XRT_HACK_UNSECURE_LOADING_XCLBIN=1
        $<TARGET_FILE:${currentTarget}>
        -x ${trace_xclbin_target}
        -i ${insts_target}
        -k MLIR_AIE
        -t ${trace_size}
        --verbosity ${verbose}
    DEPENDS ${currentTarget} trace_xclbin
    COMMENT "Running ${currentTarget} test with tracing"
)

# Parse trace target
add_custom_target(parse_trace
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/../../utils/parse_trace.py
        --input trace.txt
        --mlir ${trace_mlir_target}
        --output trace_softmax.json
    COMMENT "Parsing trace output"
)

# Clean targets
add_custom_target(clean_all
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}
    COMMENT "Cleaning all build artifacts"
)

# Display configuration
message(STATUS "Configuration:")
message(STATUS "  Vector size: ${N}")
message(STATUS "  Tile size: ${n}")
message(STATUS "  Number of cores: ${n_cores}")
message(STATUS "  Data types: ${dtype_in} -> ${dtype_out}")
message(STATUS "  Device: ${devicename}")
message(STATUS "  Target: ${TARGET_NAME}")
message(STATUS "  Using PyTorch golden reference: YES")

# Display configuration
message(STATUS "Configuration:")
message(STATUS "  Vector size: ${N}")
message(STATUS "  Tile size: ${n}")
message(STATUS "  Number of cores: ${n_cores}")
message(STATUS "  Data types: ${dtype_in} -> ${dtype_out}")
message(STATUS "  Device: ${devicename}")
message(STATUS "  Target: ${TARGET_NAME}")
message(STATUS "  Using PyTorch golden reference: YES")
