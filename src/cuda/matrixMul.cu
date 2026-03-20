/**
 * @file matrixMul.cu
 * @brief Matrix Multiplication: C = A * B.
 * Optimized version using Shared Memory Tiling.
 * Parallel Computing - 2024/2025
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// Configuration Constants
#define MATRIX_DIM   1024   // Use larger dimensions to see real GPU power
#define TILE_SIZE    32     // Optimal for modern NVIDIA architectures
#define EPSILON      1e-2f  // Tolerance for float comparison

// Error checking macro
#define CHECK_CUDA(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

/**
 * Optimized Tiled Matrix Multiplication Kernel.
 * Each block computes one TILE_SIZE x TILE_SIZE sub-matrix of C.
 */
__global__ void matrixMulTiledKernel(float *C, const float *A, const float *B, int n) {
    // Shared memory for tiles of A and B
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // Identify the row and column of the C element to be computed
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float p_value = 0;

    // Loop over the A and B matrices in tiles
    for (int m = 0; m < (n / TILE_SIZE); ++m) {
        
        // Load tiles from Global Memory to Shared Memory
        s_A[ty][tx] = A[row * n + (m * TILE_SIZE + tx)];
        s_B[ty][tx] = B[(m * TILE_SIZE + ty) * n + col];

        // Synchronize to ensure the tile is fully loaded
        __syncthreads();

        // Compute the partial dot product for this tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            p_value += s_A[ty][k] * s_B[k][tx];
        }

        // Synchronize before loading the next tile
        __syncthreads();
    }

    // Write the final result to Global Memory
    if (row < n && col < n) {
        C[row * n + col] = p_value;
    }
}

// --------------------- 
// Host Utility Routines 
// --------------------- 

/**
 * CPU Reference implementation for validation.
 */
void matrixMulCPU(const float *A, const float *B, float *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

/**
 * Validates GPU results against CPU results.
 */
bool validateResults(float *h_C, float *d_C, int n) {
    for (int i = 0; i < n * n; i++) {
        if (fabsf(h_C[i] - d_C[i]) > EPSILON) {
            printf("Mismatch at index %d: CPU=%.4f GPU=%.4f\n", i, h_C[i], d_C[i]);
            return false;
        }
    }
    return true;
}

// ------------ 
// Main Program 
// ------------ 
int main() {
    int n = MATRIX_DIM;
    size_t size = n * n * sizeof(float);

    printf("Starting Matrix Multiplication: %d x %d\n", n, n);

    // Allocate Host Memory (Pinned for faster transfer)
    float *h_A, *h_B, *h_C, *h_gpu_result;
    CHECK_CUDA(cudaMallocHost((void**)&h_A, size));
    CHECK_CUDA(cudaMallocHost((void**)&h_B, size));
    CHECK_CUDA(cudaMallocHost((void**)&h_C, size));
    CHECK_CUDA(cudaMallocHost((void**)&h_gpu_result, size));

    // Initialize random data
    for (int i = 0; i < n * n; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }

    // Allocate Device Memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void**)&d_A, size));
    CHECK_CUDA(cudaMalloc((void**)&d_B, size));
    CHECK_CUDA(cudaMalloc((void**)&d_C, size));

    // Performance Measurement
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Copy to Device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Define Grid and Block dimensions
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);

    printf("Launching Tiled Kernel... Grid: (%d,%d) Block: (%d,%d)\n", 
            blocksPerGrid.x, blocksPerGrid.y, threadsPerBlock.x, threadsPerBlock.y);

    cudaEventRecord(start);
    matrixMulTiledKernel<<<blocksPerGrid, threadsPerBlock>>>(d_C, d_A, d_B, n);
    cudaEventRecord(stop);
    
    // Copy result back to Host
    CHECK_CUDA(cudaMemcpy(h_gpu_result, d_C, size, cudaMemcpyDeviceToHost));

    // Calculate time
    float milliseconds = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Validation
    printf("Computing CPU reference for validation...\n");
    matrixMulCPU(h_A, h_B, h_C, n);
    
    if (validateResults(h_C, h_gpu_result, n)) {
        printf("SUCCESS: Result matches CPU implementation!\n");
        double flops = 2.0 * n * n * n;
        double gflops = (flops * 1.0e-9) / (milliseconds * 1.0e-3);
        printf("Performance: %.2f GFLOPS | Time: %.2f ms\n", gflops, milliseconds);
    } else {
        printf("FAILURE: Results do not match.\n");
    }

    // Cleanup
    cudaFreeHost(h_A); cudaFreeHost(h_B); cudaFreeHost(h_C); cudaFreeHost(h_gpu_result);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}