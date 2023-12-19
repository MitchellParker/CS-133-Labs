#include "lib/cnn-krnl.h"

#define I_BLOCK_SIZE (32)
#define J_BLOCK_SIZE (1)
#define H_BLOCK_SIZE (1)
#define W_BLOCK_SIZE (1)

#pragma ACCEL kernel
void CnnKernel(
    const input_t input[kNum][kInImSize][kInImSize],
    const weight_t weight[kNum][kNum][kKernel][kKernel],
    const bias_t bias[kNum],
    output_t output[kNum][kOutImSize][kOutImSize]
  ) {

  // Allocate memory on heap to avoid stack overflow.
  static compute_t C[kNum][kImSize][kImSize];

  for (int i = 0; i < kNum; ++i) {
    for (int h = 0; h < kImSize; ++h) {
      for (int w = 0; w < kImSize; ++w) {
        C[i][h][w] = bias[i];
      }
    }
  }

  // Convolution
  for (int i = 0; i < kNum/I_BLOCK_SIZE; ++i) {
    #pragma ACCEL pipeline
    for (int j = 0; j < kNum/J_BLOCK_SIZE; ++j) {

      float weight_cached[I_BLOCK_SIZE][J_BLOCK_SIZE][kKernel][kKernel];

      // cache weights
      #pragma ACCEL parallel
      for (int ii = 0; ii < I_BLOCK_SIZE; ++ii) {
        #pragma ACCEL parallel
        for (int jj = 0; jj < J_BLOCK_SIZE; ++jj) {
          #pragma ACCEL parallel
          for (int p = 0; p < kKernel; ++p) {
            #pragma ACCEL parallel
            for (int q = 0; q < kKernel; ++q) {
              weight_cached[ii][jj][p][q] = weight[i*I_BLOCK_SIZE + ii][j*J_BLOCK_SIZE + jj][p][q];
            } // q
          } // p
        } // jj
      } // ii

      for (int h = 0; h < kImSize/H_BLOCK_SIZE; ++h) {
        #pragma ACCEL pipeline
        for (int w = 0; w < kImSize/W_BLOCK_SIZE; ++w) {

          float input_cached[J_BLOCK_SIZE][H_BLOCK_SIZE + kKernel][W_BLOCK_SIZE + kKernel];
          float C_cached[I_BLOCK_SIZE][H_BLOCK_SIZE][W_BLOCK_SIZE];

          // cache inputs
          #pragma ACCEL parallel
          for (int jj = 0; jj < J_BLOCK_SIZE; ++jj) {
            #pragma ACCEL parallel
            for (int hh = 0; hh < H_BLOCK_SIZE + kKernel - 1; ++hh) {
              #pragma ACCEL parallel
              for (int ww = 0; ww < W_BLOCK_SIZE + kKernel - 1; ++ww) {
                input_cached[jj][hh][ww] = input[j*J_BLOCK_SIZE + jj][h*H_BLOCK_SIZE + hh][w*W_BLOCK_SIZE + ww];
              } // ww
            } // hh
          } // jj

          // cache C
          #pragma ACCEL parallel
          for (int ii = 0; ii < I_BLOCK_SIZE; ++ii) {
            #pragma ACCEL parallel
            for (int hh = 0; hh < H_BLOCK_SIZE; ++hh) {
              #pragma ACCEL parallel
              for (int ww = 0; ww < W_BLOCK_SIZE; ++ww) {
                C_cached[ii][hh][ww] = C[i*I_BLOCK_SIZE + ii][h*H_BLOCK_SIZE + hh][w*W_BLOCK_SIZE + ww];
              } // ww
            } // hh
          } // ii

          // compute with cached values
          #pragma ACCEL parallel
          for (int ii = 0; ii < I_BLOCK_SIZE; ++ii) {
            #pragma ACCEL parallel
            for (int jj = 0; jj < J_BLOCK_SIZE; ++jj) {
              #pragma ACCEL parallel
              for (int hh = 0; hh < H_BLOCK_SIZE; ++hh) {
                #pragma ACCEL parallel
                for (int ww = 0; ww < W_BLOCK_SIZE; ++ww) {
                  #pragma ACCEL parallel
                  for (int p = 0; p < kKernel; ++p) {
                    #pragma ACCEL parallel
                    for (int q = 0; q < kKernel; ++q) {
                      C_cached[ii][hh][ww] += weight_cached[ii][jj][p][q] * input_cached[jj][hh + p][ww + q];
                    } // q
                  } // p
                } // ww
              } // hh
            } // jj
          } // ii

          // write finished C value
          #pragma ACCEL parallel
          for (int ii = 0; ii < I_BLOCK_SIZE; ++ii) {
            #pragma ACCEL parallel
            for (int hh = 0; hh < H_BLOCK_SIZE; ++hh) {
              #pragma ACCEL parallel
              for (int ww = 0; ww < W_BLOCK_SIZE; ++ww) {
                C[i*I_BLOCK_SIZE + ii][h*H_BLOCK_SIZE + hh][w*W_BLOCK_SIZE + ww] = C_cached[ii][hh][ww];
              } // ww
            } // hh
          } // ii

        } // w
      } // h
    } // j
  } // i

  // Max pooling and ReLU
  for (int i = 0; i < kNum; ++i) {
    for (int h = 0; h < kOutImSize; ++h) {
      for (int w = 0; w < kOutImSize; ++w) {
        output[i][h][w] = max(0, max(
            max(C[i][h * 2][w * 2    ], C[i][h * 2 + 1][w * 2    ]),
            max(C[i][h * 2][w * 2 + 1], C[i][h * 2 + 1][w * 2 + 1])));
      }
    }
  }
}