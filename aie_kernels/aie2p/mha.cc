//===- mha.cc ---------------------------*- C++-----*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===-----------------------------------------------------===//

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {
    void matmul_bf16_bf16(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out);
    void matmul_bf16_bf16_rowmaj(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out);
    void partial_softmax_bf16(bfloat16 *input, bfloat16 *output, bfloat16 *scale_buffer, const int32_t input_size, const int32_t row_idx, const int32_t row_size);
    void passThroughLine(int16_t *in, int16_t *out, int32_t lineWidth);


    void matmul_PV(bfloat16 *Q, bfloat16 *K, bfloat16 *out, bfloat16 *scale_buffer, const int32_t S_q, const int32_t S_kv, int32_t first_iter) {
        
        // VJUNG: Scale O_{i-1} by 1/exp(m_{i-1} - m_{i}) store in scale_buffer[3*S_kv:3*S_kv + S_kv]
        // VJUNG: Skip this for the first iteration as 1/exp(m_{i-1} - m_{i}) degenerates to inf due to m intizalized to -inf
        if (first_iter != 0) {
            for(int32_t l = 0; l < 4; l++){
                for(int32_t k = 0; k < 4; k++){ // Iterate for 4 rows 
                    for(int32_t j = 0; j < 2; j++){ // Each row is broken down into 2 blocks of 8
                        for (int32_t i = 0; i < 8; i++) {
                            out[i + j*32 + k*8 + l*64] = out[i + j*32 + k*8 + l*64] * scale_buffer[3*S_kv + (k + l*4)];
                        }
                    }
                }
            }
        }
        
        matmul_bf16_bf16_rowmaj(Q, K, out);
        
        ///// Debugging code /////
        // VJUNG: Use this to get softmax values for debugging
        // passThroughLine((int16_t*)Q, (int16_t*)out, rows*cols);
        
        // Test values in scale buffer to check that they are valid
        // out[0] = scale_buffer[0];
        // out[1] = scale_buffer[32];

        // Test that exp(-inf) is 0
        // auto vect_out = aie::begin_restrict_vector<16>((bfloat16 *)out);        
        // aie::accum<accfloat, 16> exp_val_accum = aie::zeros<accfloat, 16>();
        // aie::vector<float, 16> vect_in = aie::broadcast<float, 16>(std::numeric_limits<bfloat16>::lowest());
        // // exp_val_accum = vect_in;
        // exp_val_accum = aie::exp2<bfloat16>(vect_in);
        // *vect_out = exp_val_accum;

    }


    void rescale_O(bfloat16 *O, bfloat16 *scale_buffer, int32_t S_kv) {
        // VJUNG: Only after all KV are processed
        // VJUNG: TODO: Make this generic for every tile size
        // VJUNG: Need to scale depending on the data layout at the output of GEMM
        // VJUNG: Scale O_{i} by 1/l_{i} 
        for(int32_t l = 0; l < 4; l++){
            for(int32_t k = 0; k < 4; k++){ // Iterate for 4 rows 
                for(int32_t j = 0; j < 2; j++){ // Each row is broken down into 2 blocks of 8
                    for (int32_t i = 0; i < 8; i++) {
                        O[i + j*32 + k*8 + l*64] = O[i + j*32 + k*8 + l*64] * aie::inv(scale_buffer[2*S_kv + (k + l*4)]);
                    }
                }
            }
        }
    }


    void partial_softmax(bfloat16 *A, bfloat16 *P, bfloat16 *scale_buffer, float inv_scale, int32_t S_q, int32_t S_kv) {

        for (int32_t i = 0; i < S_q * S_kv; i++) {
            A[i] = A[i] * bfloat16(inv_scale);
        }
        for (int32_t i = 0; i < S_q; i++) {
            partial_softmax_bf16(A + S_kv*i, P + S_kv*i, scale_buffer, S_kv, i, S_q);
        }
    }

    void init_scale_buffer(bfloat16 *scale_buffer, int32_t size) {
        // VJUNG: TODO: Vectorize

        // VJUNG: m_{i-1} vector
        for (int32_t i = 0; i < size; i++) {
            scale_buffer[i] = bfloat16(std::numeric_limits<bfloat16>::lowest());
        }
        // VJUNG: m_{i} vector
        for (int32_t i = 0; i < size; i++) {
            scale_buffer[i + size] = bfloat16(std::numeric_limits<bfloat16>::lowest());
        }
        // VJUNG: l_{i} vector
        for (int32_t i = 0; i < size; i++) {
            scale_buffer[i + 2*size] = 0.0f;
        }
    }
}