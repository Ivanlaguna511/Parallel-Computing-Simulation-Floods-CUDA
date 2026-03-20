/**
 * @file transpose.cu
 * @brief Optimized Matrix Transpose using CUDA Shared Memory.
 * * This program demonstrates how to use Shared Memory to achieve coalesced 
 * memory access during a matrix transpose, while avoiding Bank Conflicts.
 */

#include <stdio.h>
#include <cuda_runtime.h>

// Configuration Constants
#define MATRIX_DIM   1024  // Size of the square matrix
#define TILE_DIM     32    // Dimension of the thread block (32x32 = 1024 threads)

/**
 * CUDA Error Checking Macro
 */
#define CHECK_CUDA(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

// -------------------- 
// Device Kernels 
// -------------------- 

/**
 * @brief Optimized Transpose Kernel.
 * * Optimization Techniques:
 * 1. Coalesced Access: Threads read a row and write a row (after transposing in shared memory).
 * 2. Shared Memory: Used as a temporary buffer to reorder data.
 * 3. Padding: 'TILE_DIM + 1' prevents Bank Conflicts when reading columns from shared memory.
 */
__global__ void transposeOptimized(float *output, const float *input, int width, int height) {
    
    // Shared memory tile with padding to avoid bank conflicts
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    // Map thread and block indices to matrix coordinates
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // 1. Read from Global Memory into Shared Memory (Coalesced Read)
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }

    // Wait for all threads in the block to finish loading the tile
    __syncthreads();

    // 2. Transpose indices for writing
    // We calculate new coordinates based on the transposed grid
    int new_x = blockIdx.y * TILE_DIM + threadIdx.x;
    int new_y = blockIdx.x * TILE_DIM + threadIdx.y;

    // 3. Write to Global Memory (Coalesced Write)
    // We read from the tile transposed: [threadIdx.x][threadIdx.y]
    if (new_x < height && new_y < width) {
        output[new_y * height + new_x] = tile[threadIdx.x][threadIdx.y];
    }
}

// --------------------- 
// Host Utility Routines 
// --------------------- 

/**
 * CPU Reference implementation for validation.
 */
void transposeCPU(float *output, float *input, int width, int height) {
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            output[i * height + j] = input[j * width + i];
        }
    }
}

/**
 * Validates GPU results against CPU results.
 */
bool validateResults(float *gpu_res, float *cpu_res, int n) {
    for (int i = 0; i < n; i++) {
        if (gpu_res[i] != cpu_res[i]) return false;
    }
    return true;
}

// ------------ 
// Main Program 
// ------------ 
int main() {
    int dim = MATRIX_DIM;
    size_t n_bytes = dim * dim * sizeof(float);

    printf("Matrix Transpose: %d x %d\n", dim, dim);

    // Allocate Host Memory
    float *h_input  = (float*)malloc(n_bytes);
    float *h_output_cpu = (float*)malloc(n_bytes);
    float *h_output_gpu = (float*)malloc(n_bytes);

    // Initialize Input Data
    for (int i = 0; i < dim * dim; i++) {
        h_input[i] = (float)i;
    }

    // Allocate Device Memory
    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc((void**)&d_input, n_bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_output, n_bytes));

    // Performance Instrumentation
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Copy data to Device
    CHECK_CUDA(cudaMemcpy(d_input, h_input, n_bytes, cudaMemcpyHostToDevice));

    // Define Execution Configuration
    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 blocks((dim + TILE_DIM - 1) / TILE_DIM, (dim + TILE_DIM - 1) / TILE_DIM);

    // Launch Optimized Kernel
    cudaEventRecord(start);
    transposeOptimized<<<blocks, threads>>>(d_output, d_input, dim, dim);
    cudaEventRecord(stop);

    // Copy result back to Host
    CHECK_CUDA(cudaMemcpy(h_output_gpu, d_output, n_bytes, cudaMemcpyDeviceToHost));

    // Performance Calculation
    float ms = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    // Validation
    printf("Validating GPU results with CPU...\n");
    transposeCPU(h_output_cpu, h_input, dim, dim);

    if (validateResults(h_output_gpu, h_output_cpu, dim * dim)) {
        float bandwidth = (2.0f * n_bytes) / (ms * 1e6f); // GB/s
        printf("SUCCESS: Result matches!\n");
        printf("Time: %.4f ms | Effective Bandwidth: %.2f GB/s\n", ms, bandwidth);
    } else {
        printf("FAILURE: Result mismatch!\n");
    }

    // Cleanup
    free(h_input); free(h_output_cpu); free(h_output_gpu);
    cudaFree(d_input); cudaFree(d_output);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}