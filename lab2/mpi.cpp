#include <cstring>
#include <cmath>
#include <iostream>

#include <mpi.h>

#include "lib/gemm.h"
#include "lib/common.h"
// You can directly use aligned_alloc
// with lab2::aligned_alloc(...)

const int I_BLOCK_SIZE = 256;
const int J_BLOCK_SIZE = 256;
const int K_BLOCK_SIZE = 8;

void GemmParallelBlocked(const float a[kI][kK], const float b[kK][kJ], float c[kI][kJ]) {

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // rows in the grid
  int rows = (int) std::sqrt(size);
  int power_of_two_rows = 1;
  while (power_of_two_rows * 2 <= rows) {
    power_of_two_rows *= 2;
  }
  rows = power_of_two_rows;

  // columns in the grid
  int cols = size / rows;

  int my_row = rank / cols;
  int my_col = rank % cols;

  // a communicator for processes in my row
  MPI_Comm row_comm;
  MPI_Comm_split(MPI_COMM_WORLD, my_row, rank, &row_comm);

  // a communicator for processes in my column
  MPI_Comm col_comm;
  MPI_Comm_split(MPI_COMM_WORLD, my_col, rank, &col_comm);

  // disperse rows of matrix a
  int local_rows = kI / rows;
  float *local_a = new float[local_rows * kK];
  {
    // scatter a vertically
    if (my_col == 0) {
      MPI_Scatter(a, local_rows * kK, MPI_FLOAT, local_a, local_rows * kK, MPI_FLOAT, 0, col_comm);
    }
    // broadcast a horizontally
    MPI_Bcast(local_a, local_rows * kK, MPI_FLOAT, 0, row_comm);
  }

  // disperse columns of matrix b
  int local_cols = kJ / cols;
  float *local_b = new float[kK * local_cols];
  {
    // scatter b horizontally
    if (my_row == 0) {
      MPI_Datatype temp, column;
      MPI_Type_vector(kK, local_cols, kJ, MPI_FLOAT, &temp);
      MPI_Type_create_resized(temp, 0, local_cols * sizeof(float), &column);
      MPI_Type_commit(&column);
      MPI_Scatter(b, 1, column, local_b, kK * local_cols, MPI_FLOAT, 0, row_comm);
      MPI_Type_free(&temp);
      MPI_Type_free(&column);
    }
    // broadcast b vertically
    MPI_Bcast(local_b, kK*local_cols, MPI_FLOAT, 0, col_comm);
  }

  float *local_c = new float[local_rows * local_cols];

  for (int i = 0; i < local_rows; i += I_BLOCK_SIZE) {
    for (int j = 0; j < local_cols; j += J_BLOCK_SIZE) {
      // Initialize the cc block with zeros
      float cc[I_BLOCK_SIZE][J_BLOCK_SIZE] = {};

      for (int k = 0; k < kK; k += K_BLOCK_SIZE) {
        // Initialize the aa and bb blocks
        float aa[I_BLOCK_SIZE][K_BLOCK_SIZE];
        float bb[K_BLOCK_SIZE][J_BLOCK_SIZE];

        // Copy data from matrix a to the aa block
        for (int ii = 0; ii < I_BLOCK_SIZE; ii++) {
          for (int kk = 0; kk < K_BLOCK_SIZE; kk++) {
            aa[ii][kk] = local_a[(i + ii) * kK + k + kk];
          }
        }

        // Copy data from matrix b to the bb block
        for (int kk = 0; kk < K_BLOCK_SIZE; kk++) {
          for (int jj = 0; jj < J_BLOCK_SIZE; jj++) {
            bb[kk][jj] = local_b[(k + kk) * local_cols + j + jj];
          }
        }

        // Multiply and accumulate into cc block
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

      // Copy the cc block back to the result matrix c
      for (int ii = 0; ii < I_BLOCK_SIZE; ii++) {
        for (int jj = 0; jj < J_BLOCK_SIZE; jj++) {
          local_c[(i + ii) * local_cols + j + jj] = cc[ii][jj];
        }
      }

    } // j loop
  } // i loop

  // assemble the pieces of c
  {
    // gather chunks into rows
    MPI_Datatype temp, chunk;
    MPI_Type_vector(local_rows, local_cols, kJ, MPI_FLOAT, &temp);
    MPI_Type_create_resized(temp, 0, local_cols * sizeof(float), &chunk);
    MPI_Type_commit(&chunk);
    float *c_buffer = new float[local_rows * kJ];
    MPI_Gather(local_c, local_rows*local_cols, MPI_FLOAT, c_buffer, 1, chunk, 0, row_comm);
    if (my_col == 0) {
      // gather rows into full matrix
      MPI_Gather(c_buffer, local_rows * kJ, MPI_FLOAT, c, local_rows * kJ, MPI_FLOAT, 0, col_comm);
    }
    MPI_Type_free(&temp);
    MPI_Type_free(&chunk);
    delete[] c_buffer;
  }

  delete[] local_a;
  delete[] local_b;
  delete[] local_c;

  MPI_Comm_free(&row_comm);
  MPI_Comm_free(&col_comm);

}
