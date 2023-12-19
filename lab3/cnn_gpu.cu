#include "lib/cnn.cuh"
#include "cnn_gpu.cuh"

__shared__ float in[4][36][36];
// __shared__ float wt[4][5][5];

__global__ void cnn_gpu(
    float* input,
    float* weight,
    float* bias,
    float* output)
{

  int ii = threadIdx.x;
  int hh = threadIdx.y * 2;
  int ww = threadIdx.z * 2;

  int i =   blockIdx.x * blockDim.x + threadIdx.x;
  int h = ( blockIdx.y * blockDim.y + threadIdx.y ) * 2;
  int w = ( blockIdx.z * blockDim.z + threadIdx.z ) * 2;


  float c[2][2];

  // Bias
  c[0][0] = c[0][1] = c[1][0] = c[1][1] = bias[i];

  // Convolution
  for (int j = 0; j < kNum; j++) {
    // cache inputs
    in[ii][hh  ][ww  ] = input(j,h  ,w  );
    in[ii][hh  ][ww+1] = input(j,h  ,w+1);
    in[ii][hh+1][ww  ] = input(j,h+1,w  );
    in[ii][hh+1][ww+1] = input(j,h+1,w+1);
    if (hh < 4 && ww >= 2*blockDim.z-4) {
      in[ii][hh  ][4+ww  ] = input(j,h  ,4+w  );
      in[ii][hh  ][4+ww+1] = input(j,h  ,4+w+1);
      in[ii][hh+1][4+ww  ] = input(j,h+1,4+w  );
      in[ii][hh+1][4+ww+1] = input(j,h+1,4+w+1);
    }
    if (hh >= 2*blockDim.y-4 && ww < 4) {
      in[ii][4+hh  ][ww  ] = input(j,4+h  ,w  );
      in[ii][4+hh  ][ww+1] = input(j,4+h  ,w+1);
      in[ii][4+hh+1][ww  ] = input(j,4+h+1,w  );
      in[ii][4+hh+1][ww+1] = input(j,4+h+1,w+1);
    }
    if (hh >= 2*blockDim.y-4 || ww >= 2*blockDim.z-4) {
      in[ii][4+hh  ][4+ww  ] = input(j,4+h  ,4+w  );
      in[ii][4+hh  ][4+ww+1] = input(j,4+h  ,4+w+1);
      in[ii][4+hh+1][4+ww  ] = input(j,4+h+1,4+w  );
      in[ii][4+hh+1][4+ww+1] = input(j,4+h+1,4+w+1);
    }
    // if (hh < kKernel && ww < kKernel) {
    //   wt[ii][hh][ww] = weight(i,j,hh,ww);
    // }
    __syncthreads();

    for (int q = 0; q < kKernel; q++) {
      for (int p = 0; p < kKernel; p++) {
        // float temp = wt[ii][p][q];
        float temp = weight(i,j,p,q);
        c[0][0] += temp * in[ii][hh  +p][ww  +q];
        c[0][1] += temp * in[ii][hh  +p][ww+1+q];
        c[1][0] += temp * in[ii][hh+1+p][ww  +q];
        c[1][1] += temp * in[ii][hh+1+p][ww+1+q];
      }
    }
    __syncthreads();
  }

  // Max pooling then ReLU
  output(i,h/2,w/2) = max(0.f, max(
    max(c[0][0], c[0][1]),
    max(c[1][0], c[1][1])
  ));

}