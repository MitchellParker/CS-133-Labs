#include <cstring>

#include "lib/gemm.h"

const int I_BLOCK_SIZE = 128;
const int J_BLOCK_SIZE = 256;
const int K_BLOCK_SIZE = 128;

void GemmParallelBlocked(const float a[kI][kK], const float b[kK][kJ],
                         float c[kI][kJ]) {

  float aa[I_BLOCK_SIZE][K_BLOCK_SIZE];
  float bb[K_BLOCK_SIZE][J_BLOCK_SIZE];
  float cc[I_BLOCK_SIZE][J_BLOCK_SIZE];

  #pragma omp parallel for private(aa, bb, cc)
  for (int i = 0; i < kI; i += I_BLOCK_SIZE) {
    for (int j = 0; j < kJ; j += J_BLOCK_SIZE) {

      // reset cc block
      std::memset(cc, 0, I_BLOCK_SIZE * J_BLOCK_SIZE * sizeof(float));

      for (int k = 0; k < kK; k += K_BLOCK_SIZE) {

        // cache aa block
        for (int ii1 = 0, ii2 = i; ii1 < I_BLOCK_SIZE; ii1++, ii2++) {
          std::memcpy(aa[ii1], a[ii2]+k, K_BLOCK_SIZE * sizeof(float));
        }

        // cache bb block
        for (int kk1 = 0, kk2 = k; kk1 < K_BLOCK_SIZE; kk1++, kk2++) {
          std::memcpy(bb[kk1], b[kk2]+j, J_BLOCK_SIZE * sizeof(float));
        }

        // multipy into cc block
        // GET READY TO UNROLL AND JAM
        for (int ii = 0; ii < I_BLOCK_SIZE; ii += 4) {
          for (int kk = 0; kk < K_BLOCK_SIZE; kk += 8) {
            for (int jj = 0; jj < J_BLOCK_SIZE; jj += 4) {
              cc[ii+0][jj+0] += aa[ii+0][kk+0] * bb[kk+0][jj+0] + aa[ii+0][kk+1] * bb[kk+1][jj+0] + aa[ii+0][kk+2] * bb[kk+2][jj+0] + aa[ii+0][kk+3] * bb[kk+3][jj+0] + aa[ii+0][kk+4] * bb[kk+4][jj+0] + aa[ii+0][kk+5] * bb[kk+5][jj+0] + aa[ii+0][kk+6] * bb[kk+6][jj+0] + aa[ii+0][kk+7] * bb[kk+7][jj+0];
              cc[ii+0][jj+1] += aa[ii+0][kk+0] * bb[kk+0][jj+1] + aa[ii+0][kk+1] * bb[kk+1][jj+1] + aa[ii+0][kk+2] * bb[kk+2][jj+1] + aa[ii+0][kk+3] * bb[kk+3][jj+1] + aa[ii+0][kk+4] * bb[kk+4][jj+1] + aa[ii+0][kk+5] * bb[kk+5][jj+1] + aa[ii+0][kk+6] * bb[kk+6][jj+1] + aa[ii+0][kk+7] * bb[kk+7][jj+1];
              cc[ii+0][jj+2] += aa[ii+0][kk+0] * bb[kk+0][jj+2] + aa[ii+0][kk+1] * bb[kk+1][jj+2] + aa[ii+0][kk+2] * bb[kk+2][jj+2] + aa[ii+0][kk+3] * bb[kk+3][jj+2] + aa[ii+0][kk+4] * bb[kk+4][jj+2] + aa[ii+0][kk+5] * bb[kk+5][jj+2] + aa[ii+0][kk+6] * bb[kk+6][jj+2] + aa[ii+0][kk+7] * bb[kk+7][jj+2];
              cc[ii+0][jj+3] += aa[ii+0][kk+0] * bb[kk+0][jj+3] + aa[ii+0][kk+1] * bb[kk+1][jj+3] + aa[ii+0][kk+2] * bb[kk+2][jj+3] + aa[ii+0][kk+3] * bb[kk+3][jj+3] + aa[ii+0][kk+4] * bb[kk+4][jj+3] + aa[ii+0][kk+5] * bb[kk+5][jj+3] + aa[ii+0][kk+6] * bb[kk+6][jj+3] + aa[ii+0][kk+7] * bb[kk+7][jj+3];
              cc[ii+1][jj+0] += aa[ii+1][kk+0] * bb[kk+0][jj+0] + aa[ii+1][kk+1] * bb[kk+1][jj+0] + aa[ii+1][kk+2] * bb[kk+2][jj+0] + aa[ii+1][kk+3] * bb[kk+3][jj+0] + aa[ii+1][kk+4] * bb[kk+4][jj+0] + aa[ii+1][kk+5] * bb[kk+5][jj+0] + aa[ii+1][kk+6] * bb[kk+6][jj+0] + aa[ii+1][kk+7] * bb[kk+7][jj+0];
              cc[ii+1][jj+1] += aa[ii+1][kk+0] * bb[kk+0][jj+1] + aa[ii+1][kk+1] * bb[kk+1][jj+1] + aa[ii+1][kk+2] * bb[kk+2][jj+1] + aa[ii+1][kk+3] * bb[kk+3][jj+1] + aa[ii+1][kk+4] * bb[kk+4][jj+1] + aa[ii+1][kk+5] * bb[kk+5][jj+1] + aa[ii+1][kk+6] * bb[kk+6][jj+1] + aa[ii+1][kk+7] * bb[kk+7][jj+1];
              cc[ii+1][jj+2] += aa[ii+1][kk+0] * bb[kk+0][jj+2] + aa[ii+1][kk+1] * bb[kk+1][jj+2] + aa[ii+1][kk+2] * bb[kk+2][jj+2] + aa[ii+1][kk+3] * bb[kk+3][jj+2] + aa[ii+1][kk+4] * bb[kk+4][jj+2] + aa[ii+1][kk+5] * bb[kk+5][jj+2] + aa[ii+1][kk+6] * bb[kk+6][jj+2] + aa[ii+1][kk+7] * bb[kk+7][jj+2];
              cc[ii+1][jj+3] += aa[ii+1][kk+0] * bb[kk+0][jj+3] + aa[ii+1][kk+1] * bb[kk+1][jj+3] + aa[ii+1][kk+2] * bb[kk+2][jj+3] + aa[ii+1][kk+3] * bb[kk+3][jj+3] + aa[ii+1][kk+4] * bb[kk+4][jj+3] + aa[ii+1][kk+5] * bb[kk+5][jj+3] + aa[ii+1][kk+6] * bb[kk+6][jj+3] + aa[ii+1][kk+7] * bb[kk+7][jj+3];
              cc[ii+2][jj+0] += aa[ii+2][kk+0] * bb[kk+0][jj+0] + aa[ii+2][kk+1] * bb[kk+1][jj+0] + aa[ii+2][kk+2] * bb[kk+2][jj+0] + aa[ii+2][kk+3] * bb[kk+3][jj+0] + aa[ii+2][kk+4] * bb[kk+4][jj+0] + aa[ii+2][kk+5] * bb[kk+5][jj+0] + aa[ii+2][kk+6] * bb[kk+6][jj+0] + aa[ii+2][kk+7] * bb[kk+7][jj+0];
              cc[ii+2][jj+1] += aa[ii+2][kk+0] * bb[kk+0][jj+1] + aa[ii+2][kk+1] * bb[kk+1][jj+1] + aa[ii+2][kk+2] * bb[kk+2][jj+1] + aa[ii+2][kk+3] * bb[kk+3][jj+1] + aa[ii+2][kk+4] * bb[kk+4][jj+1] + aa[ii+2][kk+5] * bb[kk+5][jj+1] + aa[ii+2][kk+6] * bb[kk+6][jj+1] + aa[ii+2][kk+7] * bb[kk+7][jj+1];
              cc[ii+2][jj+2] += aa[ii+2][kk+0] * bb[kk+0][jj+2] + aa[ii+2][kk+1] * bb[kk+1][jj+2] + aa[ii+2][kk+2] * bb[kk+2][jj+2] + aa[ii+2][kk+3] * bb[kk+3][jj+2] + aa[ii+2][kk+4] * bb[kk+4][jj+2] + aa[ii+2][kk+5] * bb[kk+5][jj+2] + aa[ii+2][kk+6] * bb[kk+6][jj+2] + aa[ii+2][kk+7] * bb[kk+7][jj+2];
              cc[ii+2][jj+3] += aa[ii+2][kk+0] * bb[kk+0][jj+3] + aa[ii+2][kk+1] * bb[kk+1][jj+3] + aa[ii+2][kk+2] * bb[kk+2][jj+3] + aa[ii+2][kk+3] * bb[kk+3][jj+3] + aa[ii+2][kk+4] * bb[kk+4][jj+3] + aa[ii+2][kk+5] * bb[kk+5][jj+3] + aa[ii+2][kk+6] * bb[kk+6][jj+3] + aa[ii+2][kk+7] * bb[kk+7][jj+3];
              cc[ii+3][jj+0] += aa[ii+3][kk+0] * bb[kk+0][jj+0] + aa[ii+3][kk+1] * bb[kk+1][jj+0] + aa[ii+3][kk+2] * bb[kk+2][jj+0] + aa[ii+3][kk+3] * bb[kk+3][jj+0] + aa[ii+3][kk+4] * bb[kk+4][jj+0] + aa[ii+3][kk+5] * bb[kk+5][jj+0] + aa[ii+3][kk+6] * bb[kk+6][jj+0] + aa[ii+3][kk+7] * bb[kk+7][jj+0];
              cc[ii+3][jj+1] += aa[ii+3][kk+0] * bb[kk+0][jj+1] + aa[ii+3][kk+1] * bb[kk+1][jj+1] + aa[ii+3][kk+2] * bb[kk+2][jj+1] + aa[ii+3][kk+3] * bb[kk+3][jj+1] + aa[ii+3][kk+4] * bb[kk+4][jj+1] + aa[ii+3][kk+5] * bb[kk+5][jj+1] + aa[ii+3][kk+6] * bb[kk+6][jj+1] + aa[ii+3][kk+7] * bb[kk+7][jj+1];
              cc[ii+3][jj+2] += aa[ii+3][kk+0] * bb[kk+0][jj+2] + aa[ii+3][kk+1] * bb[kk+1][jj+2] + aa[ii+3][kk+2] * bb[kk+2][jj+2] + aa[ii+3][kk+3] * bb[kk+3][jj+2] + aa[ii+3][kk+4] * bb[kk+4][jj+2] + aa[ii+3][kk+5] * bb[kk+5][jj+2] + aa[ii+3][kk+6] * bb[kk+6][jj+2] + aa[ii+3][kk+7] * bb[kk+7][jj+2];
              cc[ii+3][jj+3] += aa[ii+3][kk+0] * bb[kk+0][jj+3] + aa[ii+3][kk+1] * bb[kk+1][jj+3] + aa[ii+3][kk+2] * bb[kk+2][jj+3] + aa[ii+3][kk+3] * bb[kk+3][jj+3] + aa[ii+3][kk+4] * bb[kk+4][jj+3] + aa[ii+3][kk+5] * bb[kk+5][jj+3] + aa[ii+3][kk+6] * bb[kk+6][jj+3] + aa[ii+3][kk+7] * bb[kk+7][jj+3];
            }
          }
        }

      } // k loop

      // flush cc block
      for (int ii1 = i, ii2 = 0; ii1 < I_BLOCK_SIZE; ii1++, ii2++) {
        std::memcpy(c[ii1]+j, cc[ii2], J_BLOCK_SIZE * sizeof(float));
      }

    } // j loop
  } // i loop

} // GemmParallelBlocked
