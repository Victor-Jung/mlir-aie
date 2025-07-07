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
    void softmax_bf16(bfloat16 *input, bfloat16 *output, int32_t input_size);
    void passThroughLine(int16_t *in, int16_t *out, int32_t lineWidth);

    void mha_bf16_bf16(bfloat16 *Q, bfloat16 *K, bfloat16 *out, int32 kernel_toggle, float inv_scale, int32 rows, int32 cols) {
        if (kernel_toggle == 1) {
            matmul_bf16_bf16(Q, K, out);
        } else if (kernel_toggle == 2) {
            for (int32_t i = 0; i < rows * cols; i++) {
                Q[i] = Q[i] * bfloat16(inv_scale);
            }
            for (int32_t i = 0; i < rows; i++) {
                softmax_bf16(Q + cols*i, out + cols*i, cols);
            }
        } else if (kernel_toggle == 3) {
            matmul_bf16_bf16_rowmaj(Q, K, out);
            // passThroughLine((int16_t*)Q, (int16_t*)out, rows*cols);
        }
    }
}