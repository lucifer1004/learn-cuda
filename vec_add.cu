#include <iostream>

#define PROJECT_NAME "learn-cuda"
#define N 64

__global__ void VecAdd(float *A, float *B, float *C) {
  int i = threadIdx.x;
  C[i] = A[i] + B[i];
}

int main() {
  float *A = new float[N];
  float *B = new float[N];
  float *C = new float[N];
  for (int i = 0; i < N; i++) {
    A[i] = i;
    B[i] = i;
  }

  float *d_A, *d_B, *d_C;

  // If we do not malloc, the results will all be zeros.
  cudaMalloc((void **)&d_A, N * sizeof(float));
  cudaMalloc((void **)&d_B, N * sizeof(float));
  cudaMalloc((void **)&d_C, N * sizeof(float));

  cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice);

  VecAdd<<<1, N>>>(d_A, d_B, d_C);
  cudaMemcpy(C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; i++) {
    std::cout << C[i] << std::endl;
  }

  delete[] A;
  delete[] B;
  delete[] C;
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return 0;
}
