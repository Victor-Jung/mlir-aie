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
    void softmax_bf16(bfloat16 *input, bfloat16 *output, bfloat16 *scale_buffer, int32_t input_size);
    void passThroughLine(int16_t *in, int16_t *out, int32_t lineWidth);


    void matmul_PV(bfloat16 *Q, bfloat16 *K, bfloat16 *out, bfloat16 *scale_buffer) {

        matmul_bf16_bf16_rowmaj(Q, K, out);
        // passThroughLine((int16_t*)Q, (int16_t*)out, rows*cols);
        
        // Test values in scale buffer to check that they stay valid
        // out[0] = scale_buffer[0];

        // Test that exp(-inf) is 0
        auto vect_out = aie::begin_restrict_vector<16>((bfloat16 *)out);        
        aie::accum<accfloat, 16> exp_val_accum = aie::zeros<accfloat, 16>();
        aie::vector<float, 16> vect_in = aie::broadcast<float, 16>(std::numeric_limits<bfloat16>::lowest());
        // exp_val_accum = vect_in;
        exp_val_accum = aie::exp2<bfloat16>(vect_in);
        *vect_out = exp_val_accum;

    }

    void partial_softmax(bfloat16 *A, bfloat16 *P, bfloat16 *scale_buffer, float inv_scale, int32_t S_q, int32_t S_kv) {

        for (int32_t i = 0; i < S_q * S_kv; i++) {
            A[i] = A[i] * bfloat16(inv_scale);
        }
        for (int32_t i = 0; i < S_q; i++) {
            softmax_bf16(A + S_kv*i, P + S_kv*i, scale_buffer, S_kv);
        }
    }
}