//===- test.cpp -------------------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "cxxopts.hpp"
#include <bits/stdc++.h>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdfloat>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "common.h"

#include "golden_reference_verification.h"

#ifndef DATATYPES_USING_DEFINED
#define DATATYPES_USING_DEFINED
#ifndef DTYPE_IN
#define DTYPE_IN std::bfloat16_t
#endif
#ifndef DTYPE_OUT
#define DTYPE_OUT std::bfloat16_t
#endif
#ifndef DTYPE_ACC
#define DTYPE_ACC float
#endif
using ACDTYPE_IN = DTYPE_ACC;
#endif

#define XSTR(X) STR(X)
#define STR(X) #X

// Verification tolerance
// See "Note on Numerical Tolerances" in README.md
float abs_tol = matmul_common::get_abs_tol<DTYPE_IN>();
float rel_tol = matmul_common::get_rel_tol<DTYPE_IN>();

int main(int argc, const char *argv[]) {
  // Program arguments parsing
  cxxopts::Options options("Matrix Matrix Multiplication Test");
  cxxopts::ParseResult vm;
  matmul_common::add_default_options(options);

  matmul_common::parse_options(argc, argv, options, vm);
  int verbosity = vm["verbosity"].as<int>();
  int do_verify = vm["verify"].as<bool>();
  int n_iterations = vm["iters"].as<int>();
  int n_warmup_iterations = vm["warmup"].as<int>();
  int trace_size = vm["trace_sz"].as<int>();

  // Fix the seed to ensure reproducibility in CI.
  srand(1726250518); // srand(time(NULL));

  int heads = vm["heads"].as<int>();
  int S_q = vm["S_q"].as<int>();
  int S_kv = vm["S_kv"].as<int>();
  int d = vm["d"].as<int>();

  int Q_VOLUME = heads * S_q * d;
  int K_VOLUME = heads * S_kv * d;
  int QK_VOLUME = heads * S_q * S_kv;

  size_t Q_SIZE = (Q_VOLUME * sizeof(DTYPE_IN));
  size_t K_SIZE = (K_VOLUME * sizeof(DTYPE_IN));
  size_t QK_SIZE = (QK_VOLUME * sizeof(DTYPE_IN));

  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());

  if (verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << "\n";

  // Start the XRT test code
  // Get a device handle
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  // Load the xclbin
  if (verbosity >= 1)
    std::cout << "Loading xclbin: " << vm["xclbin"].as<std::string>() << "\n";
  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());

  if (verbosity >= 1)
    std::cout << "Kernel opcode: " << vm["kernel"].as<std::string>() << "\n";
  std::string Node = vm["kernel"].as<std::string>();

  // Get the kernel from the xclbin
  auto xkernels = xclbin.get_kernels();
  auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                               [Node, verbosity](xrt::xclbin::kernel &k) {
                                 auto name = k.get_name();
                                 if (verbosity >= 1) {
                                   std::cout << "Name: " << name << std::endl;
                                 }
                                 return name.rfind(Node, 0) == 0;
                               });
  auto kernelName = xkernel.get_name();

  if (verbosity >= 1)
    std::cout << "Registering xclbin: " << vm["xclbin"].as<std::string>()
              << "\n";

  device.register_xclbin(xclbin);

  // get a hardware context
  if (verbosity >= 1)
    std::cout << "Getting hardware context.\n";
  xrt::hw_context context(device, xclbin.get_uuid());

  // get a kernel handle
  if (verbosity >= 1)
    std::cout << "Getting handle to kernel:" << kernelName << "\n";
  auto kernel = xrt::kernel(context, kernelName);

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_q =
      xrt::bo(device, Q_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_k =
      xrt::bo(device, K_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_qk =
      xrt::bo(device, QK_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
  auto bo_v =
        xrt::bo(device, K_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(6));
  auto bo_a =
        xrt::bo(device, QK_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(7));

  // Workaround so we declare a really small trace buffer when one is not used
  int tmp_trace_size = (trace_size > 0) ? trace_size : 1;
  auto bo_trace = xrt::bo(device, tmp_trace_size * 4, XRT_BO_FLAGS_HOST_ONLY,
                          kernel.group_id(7));

  if (verbosity >= 1) {
    std::cout << "Writing data into buffer objects.\n";
  }

  DTYPE_IN *bufQ = bo_q.map<DTYPE_IN *>();
  std::vector<DTYPE_IN> QVec(Q_VOLUME);
  DTYPE_IN *bufK = bo_k.map<DTYPE_IN *>();
  std::vector<DTYPE_IN> KVec(K_VOLUME);
  DTYPE_IN *bufV = bo_v.map<DTYPE_IN *>();
  std::vector<DTYPE_IN> VVec(K_VOLUME);
  
  // Load input data from golden reference for consistency
  golden_reference_verification::load_golden_inputs(QVec, KVec, VVec);
  if (verbosity >= 1) {
    golden_reference_verification::print_golden_reference_info();
    std::cout << "Loaded golden reference inputs:" << std::endl;
    std::cout << "  A[0] = " << (int)QVec[0] << ", A[1] = " << (int)QVec[1] << std::endl;
    std::cout << "  B[0] = " << (int)KVec[0] << ", B[1] = " << (int)KVec[1] << std::endl;
  }

  memcpy(bufQ, QVec.data(), (QVec.size() * sizeof(DTYPE_IN)));
  memcpy(bufK, KVec.data(), (KVec.size() * sizeof(DTYPE_IN)));
  memcpy(bufV, VVec.data(), (VVec.size() * sizeof(DTYPE_IN)));  

  // Initialize outputs; bufQK is results matrix plus tracing info
  char *bufQK = bo_qk.map<char *>();
  std::vector<DTYPE_IN> QKVec(QK_VOLUME);
  memset(bufQK, 0, QK_SIZE);

  char *bufA = bo_a.map<char *>();
  std::vector<DTYPE_IN> AVec(QK_VOLUME);
  memset(bufA, 0, QK_SIZE);

  char *bufTrace = bo_trace.map<char *>();
  if (trace_size > 0)
    memset(bufTrace, 0, trace_size);

  if (verbosity >= 2) {
    std::cout << "DTYPE_IN  = " XSTR(DTYPE_IN) "\n";
    std::cout << "DTYPE_OUT = " XSTR(DTYPE_OUT) "\n";
    std::cout << "Verification tolerance " << abs_tol << " absolute, "
              << rel_tol << " relative.\n";
    std::cout << "A = \n";
    matmul_common::print_matrix(QVec, d);
    std::cout << "B = \n";
    matmul_common::print_matrix(KVec, S_kv);
  }

  // Instruction buffer for DMA configuration
  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_q.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_k.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_v.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_qk.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  
  if (trace_size > 0)
    bo_trace.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned num_iter = n_iterations + n_warmup_iterations;
  float npu_time_total = 0;
  float npu_time_min = 9999999;
  float npu_time_max = 0;

  int errors = 0;
  float macs = 2.0 * float(heads) * float(S_q) * float(S_kv) * float(d);

  for (unsigned iter = 0; iter < num_iter; iter++) {

    if (verbosity >= 1) {
      std::cout << "Running Kernel (iteration " << iter << ").\n";
    }
    auto start = std::chrono::high_resolution_clock::now();
    unsigned int opcode = 3;
    auto run = kernel(opcode, bo_instr, instr_v.size(), bo_q, bo_k, bo_v, bo_qk, bo_trace);
    ert_cmd_state r = run.wait();
    if (r != ERT_CMD_STATE_COMPLETED) {
      std::cout << "Kernel did not complete. Returned status: " << r << "\n";
      return 1;
    }
    auto stop = std::chrono::high_resolution_clock::now();
    bo_qk.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    if (trace_size > 0)
      bo_trace.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    if (iter < n_warmup_iterations) {
      /* Warmup iterations do not count towards average runtime. */
      continue;
    }

    if (do_verify) {
      memcpy(QKVec.data(), bufQK, (QKVec.size() * sizeof(DTYPE_IN)));
      if (verbosity >= 1) {
          std::cout << "Verifying against PyTorch golden reference..." << std::endl;
        }
      auto vstart = std::chrono::system_clock::now();
      
      errors = golden_reference_verification::verify_against_golden<DTYPE_IN, DTYPE_IN, ACDTYPE_IN>(
          QKVec, verbosity, abs_tol, rel_tol);
      
      auto vstop = std::chrono::system_clock::now();
      float vtime =
          std::chrono::duration_cast<std::chrono::seconds>(vstop - vstart)
              .count();
      if (verbosity >= 1) {
        std::cout << "Verify time: " << vtime << " s." << std::endl;
      }
    } else {
      if (verbosity >= 1)
        std::cout << "WARNING: matmul results not verified." << std::endl;
    }

    float npu_time =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
            .count();

    npu_time_total += npu_time;
    npu_time_min = (npu_time < npu_time_min) ? npu_time : npu_time_min;
    npu_time_max = (npu_time > npu_time_max) ? npu_time : npu_time_max;
  }

  // Only write out trace of last iteration.
  if (trace_size > 0) {
    matmul_common::write_out_trace((char *)bufTrace, trace_size,
                                   vm["trace_file"].as<std::string>());
  }

  std::cout << std::endl
            << "Avg NPU matmul time: " << npu_time_total / n_iterations << "us."
            << std::endl;
  std::cout << "Avg NPU gflops: "
            << macs / (1000 * npu_time_total / n_iterations) << std::endl;

  std::cout << std::endl
            << "Min NPU matmul time: " << npu_time_min << "us." << std::endl;
  std::cout << "Max NPU gflops: " << macs / (1000 * npu_time_min) << std::endl;

  std::cout << std::endl
            << "Max NPU matmul time: " << npu_time_max << "us." << std::endl;
  std::cout << "Min NPU gflops: " << macs / (1000 * npu_time_max) << std::endl;

  if (!errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  } else {
    std::cout << "\nError count: " << errors;
    std::cout << "\n\n";

    std::cout << "\nFailed.\n\n";
    return 1;
  }
}
