//===- softmax.cc --------------------------------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===-------------------------------------------------- --------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#define SM_VEC_LEN 16   // 32
#define log2e 1.4453125 // 1.44269504089

using namespace aie;

void softmax_simple_bf16(bfloat16 *restrict input_vector,
                         bfloat16 *restrict output_vector,
                         bfloat16 *restrict scale_buffer,
                         const int32_t vector_size) {
  event0();

  // VJUNG: We do 3 passes on the vector:
  // 1. Find the max value scaled by log2e in the vector
  // 2. Calculate the exponentials of the scaled values minus the maximum
  // 3. Calculate the softmax by dividing each exponential by the sum of all exponentials
  // Note: The multiplication by log2e is very sensitive, casting it to bf16 before exponentiation leads to wrong output.
  
  auto it_log_in = aie::cbegin_restrict_vector<SM_VEC_LEN>((bfloat16 *)input_vector);
  auto it_log_out = aie::begin_restrict_vector<SM_VEC_LEN>((bfloat16 *)input_vector);
  auto it_exp_in = aie::cbegin_restrict_vector<SM_VEC_LEN>((bfloat16 *)input_vector);
  auto it_exp_out = aie::begin_restrict_vector<SM_VEC_LEN>((bfloat16 *)output_vector);
  auto it_scale = aie::cbegin_restrict_vector<SM_VEC_LEN>((bfloat16 *)output_vector);
  auto it_soft_out = aie::begin_restrict_vector<SM_VEC_LEN>((bfloat16 *)output_vector);
  
  aie::vector<bfloat16, SM_VEC_LEN> in_elems, exp_val, input_bf16, log2e_vec, max_val_vec;
  aie::accum<accfloat, SM_VEC_LEN> out_vals, exp_val_accum, scaled_accum, exp_in_accum;

  float max_val = 0;
  float accum_exp_val = 0;
  float running_max = 0;
  bfloat16 col_sum_inv;
  const int elem_iters = vector_size / SM_VEC_LEN;

  exp_val_accum = aie::zeros<accfloat, SM_VEC_LEN>();
  
  log2e_vec = aie::broadcast<bfloat16, SM_VEC_LEN>((bfloat16)log2e);

  // First pass
  for (int i = 0; i < elem_iters; i++) {
    input_bf16 = *it_log_in++;
    scaled_accum = aie::mul(input_bf16, log2e_vec);
    running_max = aie::reduce_max(scaled_accum.to_vector<bfloat16>());
    if (running_max > max_val) {
      max_val = running_max;
    }
  }
  max_val_vec = aie::broadcast<bfloat16, SM_VEC_LEN>(max_val);
    
  // Second pass
  for (int i = 0; i < elem_iters; i++) {
    
    input_bf16 = *it_exp_in++;

    scaled_accum = aie::mul(input_bf16, log2e_vec);
    exp_in_accum = aie::sub(scaled_accum, max_val_vec);
    exp_val = aie::exp2<bfloat16>(exp_in_accum.to_vector<float>());
    exp_val_accum = add(exp_val_accum, exp_val);
    
    *it_exp_out++ = exp_val;
  }

  // Final pass
  aie::vector<float, SM_VEC_LEN> reduce = exp_val_accum.to_vector<float>();
  accum_exp_val = aie::reduce_add(reduce);
  col_sum_inv = (bfloat16)aie::inv(accum_exp_val);

  for (int c = 0; c < elem_iters; c++) {
    in_elems = *it_scale++;
    out_vals = aie::mul(in_elems, col_sum_inv);
    *it_soft_out++ = out_vals.to_vector<bfloat16>();
  }

  event1();

  return;
}


void partial_softmax_alias_bf16(bfloat16 *restrict input_vector,
                         bfloat16 *restrict output_vector,
                         bfloat16 *restrict scale_buffer,
                         const int32_t vector_size,
                         const int32_t row_idx,
                         const int32_t row_size) {
  event0();

  // VJUNG: We do 3 passes on the vector:
  // 1. Find the max value scaled by log2e in the vector
  // 2. Calculate the exponentials of the scaled values minus the maximum
  // 3. Calculate the softmax by dividing each exponential by the sum of all exponentials
  // Note: The multiplication by log2e is very sensitive, casting it to bf16 before exponentiation leads to wrong output.
  
  auto it_log_in = aie::cbegin_restrict_vector<SM_VEC_LEN>((bfloat16 *)input_vector);
  auto it_log_out = aie::begin_restrict_vector<SM_VEC_LEN>((bfloat16 *)input_vector);
  auto it_exp_in = aie::cbegin_restrict_vector<SM_VEC_LEN>((bfloat16 *)input_vector);
  auto it_exp_out = aie::begin_restrict_vector<SM_VEC_LEN>((bfloat16 *)output_vector);
  auto it_scale = aie::cbegin_restrict_vector<SM_VEC_LEN>((bfloat16 *)output_vector);
  auto it_soft_out = aie::begin_restrict_vector<SM_VEC_LEN>((bfloat16 *)output_vector);
  
  aie::vector<bfloat16, SM_VEC_LEN> in_elems, exp_val, input_bf16, log2e_vec, max_val_vec;
  aie::accum<accfloat, SM_VEC_LEN> out_vals, exp_val_accum, scaled_accum, exp_in_accum;

  float max_val = 0;
  float accum_exp_val = 0;
  float running_max = 0;
  bfloat16 col_sum_inv;
  const int elem_iters = vector_size / SM_VEC_LEN;

  exp_val_accum = aie::zeros<accfloat, SM_VEC_LEN>();
  
  log2e_vec = aie::broadcast<bfloat16, SM_VEC_LEN>((bfloat16)log2e);

  // First pass
  for (int i = 0; i < elem_iters; i++) {
    input_bf16 = *it_log_in++;
    scaled_accum = aie::mul(input_bf16, log2e_vec);
    running_max = aie::reduce_max(scaled_accum.to_vector<bfloat16>());
    if (running_max > max_val) {
      max_val = running_max;
    }
  }
  max_val_vec = aie::broadcast<bfloat16, SM_VEC_LEN>(max_val);

  // Compute m_{i}
  if (running_max > scale_buffer[row_idx]) {
    scale_buffer[row_size + row_idx] = running_max;
  }
  else{
    scale_buffer[row_size + row_idx] = scale_buffer[row_idx];
  }
    
  // Second pass
  for (int i = 0; i < elem_iters; i++) {
    
    input_bf16 = *it_exp_in++;

    scaled_accum = aie::mul(input_bf16, log2e_vec);
    exp_in_accum = aie::sub(scaled_accum, max_val_vec);
    exp_val = aie::exp2<bfloat16>(exp_in_accum.to_vector<float>());
    exp_val_accum = add(exp_val_accum, exp_val);
    
    *it_exp_out++ = exp_val;
  }


  // Final pass
  aie::vector<float, SM_VEC_LEN> reduce = exp_val_accum.to_vector<float>();
  accum_exp_val = aie::reduce_add(reduce);
  col_sum_inv = (bfloat16)aie::inv(accum_exp_val);

  for (int c = 0; c < elem_iters; c++) {
    in_elems = *it_scale++;
    // out_vals = aie::mul(in_elems, col_sum_inv);
    *it_soft_out++ = in_elems; //out_vals.to_vector<bfloat16>();
  }

  aie::accum<accfloat, 16> l_i_accum = aie::zeros<accfloat, 16>();
  aie::vector<float, 16> vect_in = aie::broadcast<float, 16>(scale_buffer[row_idx] - scale_buffer[row_size + row_idx]);
  l_i_accum = aie::exp2<bfloat16>(vect_in);
  *it_log_out = l_i_accum.to_vector<bfloat16>();

  // Store l_{i}
  scale_buffer[2*row_size + row_idx] = input_vector[0]*scale_buffer[2*row_size + row_idx] + (bfloat16)accum_exp_val;
  // Store exp(m_{i-1} - m_{i}) for next step
  scale_buffer[3*row_size + row_idx] = input_vector[0];
  // Store the current m_{i} to become m_{i-1} in the next step
  scale_buffer[row_idx] = scale_buffer[row_size + row_idx];


  event1();

  return;
}


extern "C" {

void softmax_bf16(bfloat16 *restrict input, 
                  bfloat16 *restrict output,
                  bfloat16 *restrict scale_buffer,
                  const int32_t input_size) {
  softmax_simple_bf16(input, output, scale_buffer, input_size);
}

void partial_softmax_bf16(bfloat16 *restrict input, 
                         bfloat16 *restrict output,
                         bfloat16 *restrict scale_buffer,
                         const int32_t input_size,
                         const int32_t row_idx,
                        const int32_t row_size) {
    partial_softmax_alias_bf16(input, output, scale_buffer, input_size, row_idx, row_size);
  }

} // extern "C"