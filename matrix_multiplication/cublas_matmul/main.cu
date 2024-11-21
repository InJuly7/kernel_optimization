#include <random>
#include <sys/time.h>
#include <cublas_v2.h>
#include <vector>
#include <algorithm>

const char* cuBLAS_get_error_enum(cublasStatus_t error);
void cuBLAS_assert(cublasStatus_t code, const char *file, int line);
void cuda_assert(cudaError_t code, const char *file, int line);
void check_result(const std::vector<float>& A, const std::vector<float>& B, const std::vector<float>& C, const int M, const int N, const int K);
void run_cuBLAS(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, const int M, const int N, const int K, const float alpha, const float beta);

#define cublasErrChk(ans) { cuBLAS_assert((ans), __FILE__, __LINE__); }
#define cudaErrChk(ans) { cuda_assert((ans), __FILE__, __LINE__); }

/*********************************************************
  * Configuration
  ********************************************************/
const int M = 1024;
const int N = 1024;
const int K = 1024;
const float alpha = 1.0f;
const float beta = 0.0f;

/*********************************************************
  * Helper functions
  ********************************************************/
inline float get_random_number() {return 1.0f*(std::rand()%11-5)/2.0f;}



/*********************************************************
  * cuBLAS interface functions
  ********************************************************/



/***************************************
  * Main function
  **************************************/
int main(int argc, char** argv) {

    /*** Program configuration ***/
    srand(0);
    printf("\n================================================\n");
    printf("cuBLAS GEMM Example for FP32 MatMul\n");
    printf(" -- GEMM : C[a, c] = alpha * A[a, b] @ B[b, c] + beta * C[a, c]\n");
    printf(" -- C[%d, %d] = %f * A[%d, %d] @ B[%d, %d] + %f * C[%d, %d]\n", M,N,1.0f,M,K,K,N,0.0f,M,N);
    printf(" -- total size of matrices : %.3f GB\n", 1.0f*(M*N+M*K+K*N)*sizeof(float)*1e-9);
    printf("================================================\n\n");

    /*** Initialize Data ***/
    std::vector<float> A(M*K);
    std::generate(A.begin(), A.end(), get_random_number);
    std::vector<float> B(K*N);
    std::generate(B.begin(), B.end(), get_random_number);
    std::vector<float> C(M*N, 0);

    /*** Run matmul ***/
    run_cuBLAS(A, B, C, M, N, K, alpha, beta);

    /*** Test result ***/
    #ifdef DEBUG_ON
    check_result(A, B, C, M, N, K);
    #endif


    /*** Finalize ***/

    return 0;
}



const char* cuBLAS_get_error_enum(cublasStatus_t error) {
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}


void cuBLAS_assert(cublasStatus_t code, const char *file, int line) {
   if (code != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr,"cuBLAS assert: %s %s %d\n", cuBLAS_get_error_enum(code), file, line);
      if (abort) exit(code);
   }
}

void cuda_assert(cudaError_t code, const char *file, int line) {
   if (code != cudaSuccess) {
      fprintf(stderr,"CUDA assert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void check_result(const std::vector<float>& A, const std::vector<float>& B, const std::vector<float>& C, const int M, const int N, const int K){

    printf("[TEST] Test start..\n");

    for (int y=0; y<M; y++) {
        for (int x=0; x<N; x++) {
            float sum = 0.0f;
            for (int k=0; k<K; k++) {
                sum += (A[y*K+k]*B[k*N+x]);
            }

            // Error tolerance
            if (C[y*N+x] >= sum+1e-5 || C[y*N+x] <= sum-1e-5) {
                printf(" -- [ERROR] C[%d,%d] = %f != gt(%f)\n", y, x, C[y*N+x], sum);
                printf(" -- test failed...!\n");
                return;
            }
        }
    }

    printf(" -- test passed !!\n");
    return;
}

void run_cuBLAS(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, const int M, const int N, const int K, const float alpha, const float beta) {

    printf("[Kernel] Run kernal\n");

    /*** Initialize device memory ***/
    float *d_A, *d_B, *d_C;
    cudaErrChk( cudaMalloc((void**)(&d_A), sizeof(float)*M*K) );
    cudaErrChk( cudaMalloc((void**)(&d_B), sizeof(float)*K*N) );
    cudaErrChk( cudaMalloc((void**)(&d_C), sizeof(float)*M*N) );
    cudaErrChk( cudaMemcpy(d_A, A.data(), sizeof(float)*M*K, cudaMemcpyHostToDevice) );
    cudaErrChk( cudaMemcpy(d_B, B.data(), sizeof(float)*K*N, cudaMemcpyHostToDevice) );
    cudaErrChk( cudaDeviceSynchronize() )


    /*** Setup cuBLAS execution handler ***/
    cublasHandle_t handle;
    cublasErrChk (cublasCreate (&handle));


    /*** Run CUDA kernel ***/
    
    // Record events for performance measurement
    cudaEvent_t start, stop;
    cudaErrChk( cudaEventCreate(&start) );
    cudaErrChk( cudaEventCreate(&stop) );
    cudaErrChk( cudaEventRecord(start, NULL) );

    // Run cuBLAS kernel
    cublasErrChk( cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N) );

    // End of events
    cudaErrChk(cudaEventRecord(stop, NULL));
    cudaErrChk(cudaEventSynchronize(stop));
    float msec_total = 0.0f;
    float gflo = 2.0f*M*N*K*1e-9; // multiply and add
    cudaErrChk(cudaEventElapsedTime(&msec_total, start, stop));
    printf(" -- elaped time: %.4f sec\n", msec_total*1e-3);
    printf(" -- gFlops : %.4f gflops\n", gflo/(msec_total*1e-3));

    cudaErrChk( cudaMemcpy(C.data(), d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost) );
    cudaErrChk( cudaDeviceSynchronize() )


    /*** Finalize ***/
    cudaErrChk( cudaFree(d_A) );
    cudaErrChk( cudaFree(d_B) );
    cudaErrChk( cudaFree(d_C) );
    cublasErrChk( cublasDestroy(handle) );
    
}

