// This program computes a simple version of matrix multiplication
// By: Nick from CoffeeBeforeArch

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>

__global__ void matrixMul(const float *a, const float *b, float *c, int M, int N, int K) {
  // Compute each thread's global row and column index
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Iterate over row, and down column
  // c[row * N + col] = 0;
  float Cvalue = 0;
  for (int k = 0; k < N; k++) {
    // Accumulate results for a single element
    Cvalue += a[row * N + k] * b[k * K + col];
  }
  c[row * K + col] = Cvalue;
}

// Check result on the CPU
 void verify_result(std::vector<float> &a, std::vector<float> &b, std::vector<float> &c, int M, int N, int K) {
   // For every row...
   for (int i = 0; i < M; i++) {
     // For every column...
     for (int k = 0; k < K; k++) {
      // For every element in the row-column pair
      float tmp = 0;
      for (int j = 0; j < N; j++) {
        // Accumulate the partial results
        tmp += a[i * N + j] * b[j * K + k];
      }
      // Check against the CPU result
      assert(tmp == c[i * K + k]);
    }
  }
}

int main() {
  // Matrix size of 1024 x 1024;
  int M = 1 << 5;
  int N = 1 << 5;
  int K = 1 << 5;

  // Size (in bytes) of matrix
  size_t MatA_bytes = M * N * sizeof(float);
  size_t MatB_bytes = N * K * sizeof(float);
  size_t MatC_bytes = M * K * sizeof(float);
  // Host vectors
  std::vector<float> h_a(M * N);
  std::vector<float> h_b(N * K);
  std::vector<float> h_c(M * K);

  // Initialize matrices
  std::generate(h_a.begin(), h_a.end(), []() { return rand() % 100; });
  std::generate(h_b.begin(), h_b.end(), []() { return rand() % 100; });

  // Allocate device memory
  float *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, MatA_bytes);
  cudaMalloc(&d_b, MatB_bytes);
  cudaMalloc(&d_c, MatC_bytes);

  // Copy data to the device
  cudaMemcpy(d_a, h_a.data(), MatA_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), MatB_bytes, cudaMemcpyHostToDevice);

  // Threads per CTA dimension
  int THREADS = 32;

  // Blocks per grid dimension (assumes THREADS divides N evenly)
  int BLOCK_X = M / THREADS;
  int BLOCK_Y = K / THREADS;
  // Use dim3 structs for block  and grid dimensions
  dim3 threads(THREADS, THREADS);
  dim3 blocks(BLOCK_X, BLOCK_Y);

  // Launch kernel
  matrixMul<<<blocks, threads>>>(d_a, d_b, d_c, M, N, K);

  // Copy back to the host
  cudaMemcpy(h_c.data(), d_c, MatC_bytes, cudaMemcpyDeviceToHost);

  // Check result
  verify_result(h_a, h_b, h_c, M, N, K);
  std::cout << "COMPLETED SUCCESSFULLY\n";

  // Free memory on device
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}
