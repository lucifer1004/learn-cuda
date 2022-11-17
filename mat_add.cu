#include <iostream>

#define PROJECT_NAME "learn-cuda"
#define N 64

__global__ void MatAdd(float *A, float *B, float *C) {
  // We must calculate the correct index when there are multiple blocks and
  // multiple threads.
  int idx = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x * blockDim.y +
            threadIdx.x * blockDim.y + threadIdx.y;
  C[idx] = A[idx] + B[idx];
}

int main() {
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

  float *A = new float[N * N];
  float *B = new float[N * N];
  float *C = new float[N * N];
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[i * N + j] = i;
      B[i * N + j] = j;
    }
  }

  float *d_A, *d_B, *d_C;

  // If we do not malloc, the results will all be zeros.
  cudaMalloc((void **)&d_A, N * N * sizeof(float));
  cudaMalloc((void **)&d_B, N * N * sizeof(float));
  cudaMalloc((void **)&d_C, N * N * sizeof(float));

  cudaMemcpy(d_A, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, N * N * sizeof(float), cudaMemcpyHostToDevice);

  MatAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C);
  cudaMemcpy(C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; i++) {
    std::cout << C[i * N + i] << std::endl;
  }

  delete[] A;
  delete[] B;
  delete[] C;
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return 0;
}
