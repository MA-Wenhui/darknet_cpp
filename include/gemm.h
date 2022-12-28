#ifndef GEMM_H
#define GEMM_H

#include <arm_neon.h>

#include "aarch64_amx.h"

#define ELELIST(a) \
  { a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a }

#define A_LIST(p)                                                       \
  const float amk##p[16] = {                                            \
      A[m * K + k + p] * ALPHA,        A[(m + 1) * K + k + p] * ALPHA,  \
      A[(m + 2) * K + k + p] * ALPHA,  A[(m + 3) * K + k + p] * ALPHA,  \
      A[(m + 4) * K + k + p] * ALPHA,  A[(m + 5) * K + k + p] * ALPHA,  \
      A[(m + 6) * K + k + p] * ALPHA,  A[(m + 7) * K + k + p] * ALPHA,  \
      A[(m + 8) * K + k + p] * ALPHA,  A[(m + 9) * K + k + p] * ALPHA,  \
      A[(m + 10) * K + k + p] * ALPHA, A[(m + 11) * K + k + p] * ALPHA, \
      A[(m + 12) * K + k + p] * ALPHA, A[(m + 13) * K + k + p] * ALPHA, \
      A[(m + 14) * K + k + p] * ALPHA, A[(m + 15) * K + k + p] * ALPHA}
void gemm_amx5(int M, int N, int K, float ALPHA, float *A, int lda, float *B,
               int ldb, float *C, int ldc) {
  // CLOCK_START("gemm_nn_amx5")
  AMX_SET();
  int m, n, k;
  /**
   C0 += A0[0] * B0
   C1 += A1[0] * B0
   C2 += A2[0] * B0
   ...
   C15 += A15[0] * B0

   C0 += A0[1] * B1
   C1 += A1[1] * B1
   C2 += A2[1] * B1
   ...
   C15 += A15[1] * B1
   .
   .
   .
   C0 += A0[15] * B15
   C1 += A1[15] * B15
   C2 += A2[15] * B15
   ...
   C15 += A15[15] * B15
   **/

  for (m = 0; m < M - M % 16; m += 16) {
    for (n = 0; n < N - N % 16; n += 16) {
      float *rC0 = C + (m + 0) * N + n;
      float *rC1 = C + (m + 1) * N + n;
      float *rC2 = C + (m + 2) * N + n;
      float *rC3 = C + (m + 3) * N + n;
      float *rC4 = C + (m + 4) * N + n;
      float *rC5 = C + (m + 5) * N + n;
      float *rC6 = C + (m + 6) * N + n;
      float *rC7 = C + (m + 7) * N + n;
      float *rC8 = C + (m + 8) * N + n;
      float *rC9 = C + (m + 9) * N + n;
      float *rC10 = C + (m + 10) * N + n;
      float *rC11 = C + (m + 11) * N + n;
      float *rC12 = C + (m + 12) * N + n;
      float *rC13 = C + (m + 13) * N + n;
      float *rC14 = C + (m + 14) * N + n;
      float *rC15 = C + (m + 15) * N + n;

      AMX_CLR();
      AMX_SET();
      for (k = 0; k < K - K % 16; k += 16) {
        A_LIST(0);
        A_LIST(1);
        A_LIST(2);
        A_LIST(3);
        A_LIST(4);
        A_LIST(5);
        A_LIST(6);
        A_LIST(7);
        A_LIST(8);
        A_LIST(9);
        A_LIST(10);
        A_LIST(11);
        A_LIST(12);
        A_LIST(13);
        A_LIST(14);
        A_LIST(15);

        const float *rB0 = B + (k + 0) * N + n;
        const float *rB1 = B + (k + 1) * N + n;
        const float *rB2 = B + (k + 2) * N + n;
        const float *rB3 = B + (k + 3) * N + n;
        const float *rB4 = B + (k + 4) * N + n;
        const float *rB5 = B + (k + 5) * N + n;
        const float *rB6 = B + (k + 6) * N + n;
        const float *rB7 = B + (k + 7) * N + n;
        const float *rB8 = B + (k + 8) * N + n;
        const float *rB9 = B + (k + 9) * N + n;
        const float *rB10 = B + (k + 10) * N + n;
        const float *rB11 = B + (k + 11) * N + n;
        const float *rB12 = B + (k + 12) * N + n;
        const float *rB13 = B + (k + 13) * N + n;
        const float *rB14 = B + (k + 14) * N + n;
        const float *rB15 = B + (k + 15) * N + n;

        AMX_LDY(PTR_ROW_FLAGS(amk0, 0, 0));
        AMX_LDX(PTR_ROW_FLAGS(rB0, 0, 0));
        AMX_MATFP((4ull << 42));  //每一个A0[i] 点乘 B0 然后累加到C0-C15
        AMX_LDY(PTR_ROW_FLAGS(amk1, 0, 0));
        AMX_LDX(PTR_ROW_FLAGS(rB1, 0, 0));
        AMX_MATFP((4ull << 42));
        AMX_LDY(PTR_ROW_FLAGS(amk2, 0, 0));
        AMX_LDX(PTR_ROW_FLAGS(rB2, 0, 0));
        AMX_MATFP((4ull << 42));
        AMX_LDY(PTR_ROW_FLAGS(amk3, 0, 0));
        AMX_LDX(PTR_ROW_FLAGS(rB3, 0, 0));
        AMX_MATFP((4ull << 42));
        AMX_LDY(PTR_ROW_FLAGS(amk4, 0, 0));
        AMX_LDX(PTR_ROW_FLAGS(rB4, 0, 0));
        AMX_MATFP((4ull << 42));
        AMX_LDY(PTR_ROW_FLAGS(amk5, 0, 0));
        AMX_LDX(PTR_ROW_FLAGS(rB5, 0, 0));
        AMX_MATFP((4ull << 42));
        AMX_LDY(PTR_ROW_FLAGS(amk6, 0, 0));
        AMX_LDX(PTR_ROW_FLAGS(rB6, 0, 0));
        AMX_MATFP((4ull << 42));
        AMX_LDY(PTR_ROW_FLAGS(amk7, 0, 0));
        AMX_LDX(PTR_ROW_FLAGS(rB7, 0, 0));
        AMX_MATFP((4ull << 42));
        AMX_LDY(PTR_ROW_FLAGS(amk8, 0, 0));
        AMX_LDX(PTR_ROW_FLAGS(rB8, 0, 0));
        AMX_MATFP((4ull << 42));
        AMX_LDY(PTR_ROW_FLAGS(amk9, 0, 0));
        AMX_LDX(PTR_ROW_FLAGS(rB9, 0, 0));
        AMX_MATFP((4ull << 42));
        AMX_LDY(PTR_ROW_FLAGS(amk10, 0, 0));
        AMX_LDX(PTR_ROW_FLAGS(rB10, 0, 0));
        AMX_MATFP((4ull << 42));
        AMX_LDY(PTR_ROW_FLAGS(amk11, 0, 0));
        AMX_LDX(PTR_ROW_FLAGS(rB11, 0, 0));
        AMX_MATFP((4ull << 42));
        AMX_LDY(PTR_ROW_FLAGS(amk12, 0, 0));
        AMX_LDX(PTR_ROW_FLAGS(rB12, 0, 0));
        AMX_MATFP((4ull << 42));
        AMX_LDY(PTR_ROW_FLAGS(amk13, 0, 0));
        AMX_LDX(PTR_ROW_FLAGS(rB13, 0, 0));
        AMX_MATFP((4ull << 42));
        AMX_LDY(PTR_ROW_FLAGS(amk14, 0, 0));
        AMX_LDX(PTR_ROW_FLAGS(rB14, 0, 0));
        AMX_MATFP((4ull << 42));
        AMX_LDY(PTR_ROW_FLAGS(amk15, 0, 0));
        AMX_LDX(PTR_ROW_FLAGS(rB15, 0, 0));
        AMX_MATFP((4ull << 42));
      }
      // C0-C15对应 z0 z4 z8 ... z60
      AMX_STZ(PTR_ROW_FLAGS(rC0, 0, 0));
      AMX_STZ(PTR_ROW_FLAGS(rC1, 4, 0));
      AMX_STZ(PTR_ROW_FLAGS(rC2, 8, 0));
      AMX_STZ(PTR_ROW_FLAGS(rC3, 12, 0));
      AMX_STZ(PTR_ROW_FLAGS(rC4, 16, 0));
      AMX_STZ(PTR_ROW_FLAGS(rC5, 20, 0));
      AMX_STZ(PTR_ROW_FLAGS(rC6, 24, 0));
      AMX_STZ(PTR_ROW_FLAGS(rC7, 28, 0));

      AMX_STZ(PTR_ROW_FLAGS(rC8, 32, 0));
      AMX_STZ(PTR_ROW_FLAGS(rC9, 36, 0));
      AMX_STZ(PTR_ROW_FLAGS(rC10, 40, 0));
      AMX_STZ(PTR_ROW_FLAGS(rC11, 44, 0));
      AMX_STZ(PTR_ROW_FLAGS(rC12, 48, 0));
      AMX_STZ(PTR_ROW_FLAGS(rC13, 52, 0));
      AMX_STZ(PTR_ROW_FLAGS(rC14, 56, 0));
      AMX_STZ(PTR_ROW_FLAGS(rC15, 60, 0));
    }
  }

  for (; m < M; ++m) {
    for (n = 0; n < N; ++n) {
      for (k = 0; k < K; ++k) {
        float A_PART = ALPHA * A[m * lda + k];
        C[m * ldc + n] += A_PART * B[k * ldb + n];
      }
    }
  }

  AMX_CLR();
  // CLOCK_END("gemm_nn_amx5")
}

void gemm_nn_neon(int M, int N, int K, float ALPHA, float *ma, int lda,
                  float *mb, int ldb, float *mc, int ldc) {
  int m, n, k;
  for (m = 0; m < M - M % 2; m += 2) {
    for (n = 0; n < N - N % 32; n += 32) {
      float32x4_t mmc0_0 = vld1q_f32(mc + (m + 0) * N + n + 0);
      float32x4_t mmc0_4 = vld1q_f32(mc + (m + 0) * N + n + 4);
      float32x4_t mmc0_8 = vld1q_f32(mc + (m + 0) * N + n + 8);
      float32x4_t mmc0_12 = vld1q_f32(mc + (m + 0) * N + n + 12);

      float32x4_t mmc0_16 = vld1q_f32(mc + (m + 0) * N + n + 16);
      float32x4_t mmc0_20 = vld1q_f32(mc + (m + 0) * N + n + 20);
      float32x4_t mmc0_24 = vld1q_f32(mc + (m + 0) * N + n + 24);
      float32x4_t mmc0_28 = vld1q_f32(mc + (m + 0) * N + n + 28);

      float32x4_t mmc1_0 = vld1q_f32(mc + (m + 1) * N + n + 0);
      float32x4_t mmc1_4 = vld1q_f32(mc + (m + 1) * N + n + 4);
      float32x4_t mmc1_8 = vld1q_f32(mc + (m + 1) * N + n + 8);
      float32x4_t mmc1_12 = vld1q_f32(mc + (m + 1) * N + n + 12);

      float32x4_t mmc1_16 = vld1q_f32(mc + (m + 1) * N + n + 16);
      float32x4_t mmc1_20 = vld1q_f32(mc + (m + 1) * N + n + 20);
      float32x4_t mmc1_24 = vld1q_f32(mc + (m + 1) * N + n + 24);
      float32x4_t mmc1_28 = vld1q_f32(mc + (m + 1) * N + n + 28);

      for (k = 0; k < K - K % 4; k += 4) {
        // const float32x4_t mma0_mk = vld1q_f32(ma + (m + 0) * K + k);
        // const float32x4_t mma1_mk = vld1q_f32(ma + (m + 1) * K + k);
        float ma_part_0[4] = {*(ma + (m + 0) * K + k) * ALPHA,
                              *(ma + (m + 0) * K + k + 1) * ALPHA,
                              *(ma + (m + 0) * K + k + 2) * ALPHA,
                              *(ma + (m + 0) * K + k + 3) * ALPHA};
        const float32x4_t mma0_mk = vld1q_f32(ma_part_0);
        float ma_part_1[4] = {*(ma + (m + 1) * K + k) * ALPHA,
                              *(ma + (m + 1) * K + k + 1) * ALPHA,
                              *(ma + (m + 1) * K + k + 2) * ALPHA,
                              *(ma + (m + 1) * K + k + 3) * ALPHA};
        const float32x4_t mma1_mk = vld1q_f32(ma_part_1);

        const float32x4_t mmb0_0 = vld1q_f32(mb + (k + 0) * N + n + 0);
        const float32x4_t mmb0_4 = vld1q_f32(mb + (k + 0) * N + n + 4);
        const float32x4_t mmb0_8 = vld1q_f32(mb + (k + 0) * N + n + 8);
        const float32x4_t mmb0_12 = vld1q_f32(mb + (k + 0) * N + n + 12);

        const float32x4_t mmb0_16 = vld1q_f32(mb + (k + 0) * N + n + 16);
        const float32x4_t mmb0_20 = vld1q_f32(mb + (k + 0) * N + n + 20);
        const float32x4_t mmb0_24 = vld1q_f32(mb + (k + 0) * N + n + 24);
        const float32x4_t mmb0_28 = vld1q_f32(mb + (k + 0) * N + n + 28);

        mmc0_0 = vmlaq_laneq_f32(mmc0_0, mmb0_0, mma0_mk, 0);
        mmc0_4 = vmlaq_laneq_f32(mmc0_4, mmb0_4, mma0_mk, 0);
        mmc0_8 = vmlaq_laneq_f32(mmc0_8, mmb0_8, mma0_mk, 0);
        mmc0_12 = vmlaq_laneq_f32(mmc0_12, mmb0_12, mma0_mk, 0);

        mmc0_16 = vmlaq_laneq_f32(mmc0_16, mmb0_16, mma0_mk, 0);
        mmc0_20 = vmlaq_laneq_f32(mmc0_20, mmb0_20, mma0_mk, 0);
        mmc0_24 = vmlaq_laneq_f32(mmc0_24, mmb0_24, mma0_mk, 0);
        mmc0_28 = vmlaq_laneq_f32(mmc0_28, mmb0_28, mma0_mk, 0);

        mmc1_0 = vmlaq_laneq_f32(mmc1_0, mmb0_0, mma1_mk, 0);
        mmc1_4 = vmlaq_laneq_f32(mmc1_4, mmb0_4, mma1_mk, 0);
        mmc1_8 = vmlaq_laneq_f32(mmc1_8, mmb0_8, mma1_mk, 0);
        mmc1_12 = vmlaq_laneq_f32(mmc1_12, mmb0_12, mma1_mk, 0);

        mmc1_16 = vmlaq_laneq_f32(mmc1_16, mmb0_16, mma1_mk, 0);
        mmc1_20 = vmlaq_laneq_f32(mmc1_20, mmb0_20, mma1_mk, 0);
        mmc1_24 = vmlaq_laneq_f32(mmc1_24, mmb0_24, mma1_mk, 0);
        mmc1_28 = vmlaq_laneq_f32(mmc1_28, mmb0_28, mma1_mk, 0);

        const float32x4_t mmb1_0 = vld1q_f32(mb + (k + 1) * N + n + 0);
        const float32x4_t mmb1_4 = vld1q_f32(mb + (k + 1) * N + n + 4);
        const float32x4_t mmb1_8 = vld1q_f32(mb + (k + 1) * N + n + 8);
        const float32x4_t mmb1_12 = vld1q_f32(mb + (k + 1) * N + n + 12);

        const float32x4_t mmb1_16 = vld1q_f32(mb + (k + 1) * N + n + 16);
        const float32x4_t mmb1_20 = vld1q_f32(mb + (k + 1) * N + n + 20);
        const float32x4_t mmb1_24 = vld1q_f32(mb + (k + 1) * N + n + 24);
        const float32x4_t mmb1_28 = vld1q_f32(mb + (k + 1) * N + n + 28);

        mmc0_0 = vmlaq_laneq_f32(mmc0_0, mmb1_0, mma0_mk, 1);
        mmc0_4 = vmlaq_laneq_f32(mmc0_4, mmb1_4, mma0_mk, 1);
        mmc0_8 = vmlaq_laneq_f32(mmc0_8, mmb1_8, mma0_mk, 1);
        mmc0_12 = vmlaq_laneq_f32(mmc0_12, mmb1_12, mma0_mk, 1);

        mmc0_16 = vmlaq_laneq_f32(mmc0_16, mmb1_16, mma0_mk, 1);
        mmc0_20 = vmlaq_laneq_f32(mmc0_20, mmb1_20, mma0_mk, 1);
        mmc0_24 = vmlaq_laneq_f32(mmc0_24, mmb1_24, mma0_mk, 1);
        mmc0_28 = vmlaq_laneq_f32(mmc0_28, mmb1_28, mma0_mk, 1);

        mmc1_0 = vmlaq_laneq_f32(mmc1_0, mmb1_0, mma1_mk, 1);
        mmc1_4 = vmlaq_laneq_f32(mmc1_4, mmb1_4, mma1_mk, 1);
        mmc1_8 = vmlaq_laneq_f32(mmc1_8, mmb1_8, mma1_mk, 1);
        mmc1_12 = vmlaq_laneq_f32(mmc1_12, mmb1_12, mma1_mk, 1);

        mmc1_16 = vmlaq_laneq_f32(mmc1_16, mmb1_16, mma1_mk, 1);
        mmc1_20 = vmlaq_laneq_f32(mmc1_20, mmb1_20, mma1_mk, 1);
        mmc1_24 = vmlaq_laneq_f32(mmc1_24, mmb1_24, mma1_mk, 1);
        mmc1_28 = vmlaq_laneq_f32(mmc1_28, mmb1_28, mma1_mk, 1);

        const float32x4_t mmb2_0 = vld1q_f32(mb + (k + 2) * N + n + 0);
        const float32x4_t mmb2_4 = vld1q_f32(mb + (k + 2) * N + n + 4);
        const float32x4_t mmb2_8 = vld1q_f32(mb + (k + 2) * N + n + 8);
        const float32x4_t mmb2_12 = vld1q_f32(mb + (k + 2) * N + n + 12);

        const float32x4_t mmb2_16 = vld1q_f32(mb + (k + 2) * N + n + 16);
        const float32x4_t mmb2_20 = vld1q_f32(mb + (k + 2) * N + n + 20);
        const float32x4_t mmb2_24 = vld1q_f32(mb + (k + 2) * N + n + 24);
        const float32x4_t mmb2_28 = vld1q_f32(mb + (k + 2) * N + n + 28);

        mmc0_0 = vmlaq_laneq_f32(mmc0_0, mmb2_0, mma0_mk, 2);
        mmc0_4 = vmlaq_laneq_f32(mmc0_4, mmb2_4, mma0_mk, 2);
        mmc0_8 = vmlaq_laneq_f32(mmc0_8, mmb2_8, mma0_mk, 2);
        mmc0_12 = vmlaq_laneq_f32(mmc0_12, mmb2_12, mma0_mk, 2);

        mmc0_16 = vmlaq_laneq_f32(mmc0_16, mmb2_16, mma0_mk, 2);
        mmc0_20 = vmlaq_laneq_f32(mmc0_20, mmb2_20, mma0_mk, 2);
        mmc0_24 = vmlaq_laneq_f32(mmc0_24, mmb2_24, mma0_mk, 2);
        mmc0_28 = vmlaq_laneq_f32(mmc0_28, mmb2_28, mma0_mk, 2);

        mmc1_0 = vmlaq_laneq_f32(mmc1_0, mmb2_0, mma1_mk, 2);
        mmc1_4 = vmlaq_laneq_f32(mmc1_4, mmb2_4, mma1_mk, 2);
        mmc1_8 = vmlaq_laneq_f32(mmc1_8, mmb2_8, mma1_mk, 2);
        mmc1_12 = vmlaq_laneq_f32(mmc1_12, mmb2_12, mma1_mk, 2);

        mmc1_16 = vmlaq_laneq_f32(mmc1_16, mmb2_16, mma1_mk, 2);
        mmc1_20 = vmlaq_laneq_f32(mmc1_20, mmb2_20, mma1_mk, 2);
        mmc1_24 = vmlaq_laneq_f32(mmc1_24, mmb2_24, mma1_mk, 2);
        mmc1_28 = vmlaq_laneq_f32(mmc1_28, mmb2_28, mma1_mk, 2);

        const float32x4_t mmb3_0 = vld1q_f32(mb + (k + 3) * N + n + 0);
        const float32x4_t mmb3_4 = vld1q_f32(mb + (k + 3) * N + n + 4);
        const float32x4_t mmb3_8 = vld1q_f32(mb + (k + 3) * N + n + 8);
        const float32x4_t mmb3_12 = vld1q_f32(mb + (k + 3) * N + n + 12);

        const float32x4_t mmb3_16 = vld1q_f32(mb + (k + 3) * N + n + 16);
        const float32x4_t mmb3_20 = vld1q_f32(mb + (k + 3) * N + n + 20);
        const float32x4_t mmb3_24 = vld1q_f32(mb + (k + 3) * N + n + 24);
        const float32x4_t mmb3_28 = vld1q_f32(mb + (k + 3) * N + n + 28);

        mmc0_0 = vmlaq_laneq_f32(mmc0_0, mmb3_0, mma0_mk, 3);
        mmc0_4 = vmlaq_laneq_f32(mmc0_4, mmb3_4, mma0_mk, 3);
        mmc0_8 = vmlaq_laneq_f32(mmc0_8, mmb3_8, mma0_mk, 3);
        mmc0_12 = vmlaq_laneq_f32(mmc0_12, mmb3_12, mma0_mk, 3);

        mmc0_16 = vmlaq_laneq_f32(mmc0_16, mmb3_16, mma0_mk, 3);
        mmc0_20 = vmlaq_laneq_f32(mmc0_20, mmb3_20, mma0_mk, 3);
        mmc0_24 = vmlaq_laneq_f32(mmc0_24, mmb3_24, mma0_mk, 3);
        mmc0_28 = vmlaq_laneq_f32(mmc0_28, mmb3_28, mma0_mk, 3);

        mmc1_0 = vmlaq_laneq_f32(mmc1_0, mmb3_0, mma1_mk, 3);
        mmc1_4 = vmlaq_laneq_f32(mmc1_4, mmb3_4, mma1_mk, 3);
        mmc1_8 = vmlaq_laneq_f32(mmc1_8, mmb3_8, mma1_mk, 3);
        mmc1_12 = vmlaq_laneq_f32(mmc1_12, mmb3_12, mma1_mk, 3);

        mmc1_16 = vmlaq_laneq_f32(mmc1_16, mmb3_16, mma1_mk, 3);
        mmc1_20 = vmlaq_laneq_f32(mmc1_20, mmb3_20, mma1_mk, 3);
        mmc1_24 = vmlaq_laneq_f32(mmc1_24, mmb3_24, mma1_mk, 3);
        mmc1_28 = vmlaq_laneq_f32(mmc1_28, mmb3_28, mma1_mk, 3);
      }

      vst1q_f32(mc + (m + 0) * N + n + 0, mmc0_0);
      vst1q_f32(mc + (m + 0) * N + n + 4, mmc0_4);
      vst1q_f32(mc + (m + 0) * N + n + 8, mmc0_8);
      vst1q_f32(mc + (m + 0) * N + n + 12, mmc0_12);

      vst1q_f32(mc + (m + 0) * N + n + 16, mmc0_16);
      vst1q_f32(mc + (m + 0) * N + n + 20, mmc0_20);
      vst1q_f32(mc + (m + 0) * N + n + 24, mmc0_24);
      vst1q_f32(mc + (m + 0) * N + n + 28, mmc0_28);

      vst1q_f32(mc + (m + 1) * N + n + 0, mmc1_0);
      vst1q_f32(mc + (m + 1) * N + n + 4, mmc1_4);
      vst1q_f32(mc + (m + 1) * N + n + 8, mmc1_8);
      vst1q_f32(mc + (m + 1) * N + n + 12, mmc1_12);

      vst1q_f32(mc + (m + 1) * N + n + 16, mmc1_16);
      vst1q_f32(mc + (m + 1) * N + n + 20, mmc1_20);
      vst1q_f32(mc + (m + 1) * N + n + 24, mmc1_24);
      vst1q_f32(mc + (m + 1) * N + n + 28, mmc1_28);
    }
  }

  for (; m < M; ++m) {
    for (n = 0; n < N; ++n) {
      for (k = 0; k < K; ++k) {
        float A_PART = ALPHA * ma[m * lda + k];
        mc[m * ldc + n] += A_PART * mb[k * ldb + n];
      }
    }
  }
}

#define CLOCK_START(name)  \
  printf("%s in\n", name); \
  clock_t start, end;      \
  double cpu_time_used;    \
  start = clock();

#define CLOCK_END(name)                                     \
  end = clock();                                            \
  cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC; \
  printf("%s time: %f sec\n", name, cpu_time_used);

#define amx_z_to_x(zrow, xrow) \
  (1ull << 63) + (8ull << 11) + (1ull << 26) + ((zrow) << 20) + 64 * (xrow)

#define ZERO_Z(row) AMX_FMA32((0ull << 63) + (7ull << 27) + (row << 20))

void gemm_nn(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    // #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
             float A_PART = ALPHA*A[i*lda+k];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    //printf("cpu: %d %d %d %d %d %f %d %d %f %d\n",TA, TB, M, N, K, ALPHA, lda, ldb, BETA, ldc);
    int i, j;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            C[i*ldc + j] *= BETA;
        }
    }
    if(!TA && !TB)
        gemm_nn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
}

void gemm(int TA, int TB, 
        int M, int N, int K, 
        float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    gemm_cpu( TA,  TB,  M, N, K, ALPHA,A,lda, B, ldb,BETA,C,ldc);
}

#endif