/**
 * @file scalarProd.cu
 * @brief Optimized scalar product (dot product) of two vectors using CUDA.
 * * Logic:
 * 1. Multiply elements A[i] * B[i] and store in shared memory.
 * 2. Perform a tree-based parallel reduction within each block.
 * 3. Sum the partial block results on the CPU.
 */

#include <stdio.h>
#include <cuda_runtime.h>

// Configuration Constants
#define N 1 << 20          // Processing 1 million elements to show GPU power
#define BLOCK_SIZE 256     // Number of threads per block

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
 * @brief Kernel that computes partial dot products.
 * * This kernel combines the multiplication and the first reduction stage.
 * It uses a "Grid-Stride Loop" to handle vectors larger than the grid.
 */
__global__ void dotProductKernel(const float *A, const float *B, float *PartialSums, int n) {
    // Dynamically allocated shared memory for this block
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int gridStride = blockDim.x * gridDim.x;

    float tempSum = 0.0f;

    // Grid-Stride Loop: Each thread processes multiple elements if N > GridSize
    while (i < n) {
        tempSum += A[i] * B[i];
        i += gridStride;
    }

    // Store the thread's local sum in shared memory
    sdata[tid] = tempSum;
    __syncthreads();

    // In-place Tree Reduction in Shared Memory
    // This loop reduces the sdata array to a single value (at sdata[0])
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // The first thread in the block writes the result to global memory
    if (tid == 0) {
        PartialSums[blockIdx.x] = sdata[0];
    }
}

// --------------------- 
// Host Utility Routines 
// --------------------- 

/**
 * @brief CPU Reference implementation for validation.
 */
double dotProductCPU(const float *A, const float *B, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += (double)(A[i] * B[i]);
    }
    return sum;
}

// ------------ 
// Main Program 
// ------------ 
int main() {
    int n_elem = N;
    size_t vec_bytes = n_elem * sizeof(float);

    // Grid calculation
    // Limit grid size to avoid launching more blocks than necessary for hardware
    int n_blocks = (n_elem + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (n_blocks > 1024) n_blocks = 1024; 
    size_t partial_bytes = n_blocks * sizeof(float);

    printf("Vector Size: %d | Blocks: %d | Threads per block: %d\n", n_elem, n_blocks, BLOCK_SIZE);

    // 1. Allocate Pinned Host Memory (faster transfers via DMA)
    float *h_A, *h_B, *h_Partial;
    CHECK_CUDA(cudaMallocHost((void**)&h_A, vec_bytes));
    CHECK_CUDA(cudaMallocHost((void**)&h_B, vec_bytes));
    CHECK_CUDA(cudaMallocHost((void**)&h_Partial, partial_bytes));

    // 2. Initialize Data
    srand(42);
    for (int i = 0; i < n_elem; i++) {
        h_A[i] = (float)rand() / (float)RAND_MAX;
        h_B[i] = (float)rand() / (float)RAND_MAX;
    }

    // 3. Allocate Device Memory
    float *d_A, *d_B, *d_Partial;
    CHECK_CUDA(cudaMalloc((void**)&d_A, vec_bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_B, vec_bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_Partial, partial_bytes));

    // 4. Performance Instrumentation
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 5. Transfer to Device and Execute
    cudaEventRecord(start);
    CHECK_CUDA(cudaMemcpy(d_A, h_A, vec_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, vec_bytes, cudaMemcpyHostToDevice));

    // Launch kernel with dynamic shared memory (size = BLOCK_SIZE * 4 bytes)
    size_t shared_mem_size = BLOCK_SIZE * sizeof(float);
    dotProductKernel<<<n_blocks, BLOCK_SIZE, shared_mem_size>>>(d_A, d_B, d_Partial, n_elem);

    // 6. Transfer results back
    CHECK_CUDA(cudaMemcpy(h_Partial, d_Partial, partial_bytes, cudaMemcpyDeviceToHost));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // 7. Final Sum on Host
    // Using double precision for the final accumulator to prevent overflow/rounding errors
    double gpu_result = 0.0;
    for (int i = 0; i < n_blocks; i++) {
        gpu_result += (double)h_Partial[i];
    }

    // 8. Validation
    float msec_total = 0.0f;
    cudaEventElapsedTime(&msec_total, start, stop);
    
    printf("Computing CPU reference...\n");
    double cpu_result = dotProductCPU(h_A, h_B, n_elem);

    // Results comparison
    double diff = fabs(cpu_result - gpu_result);
    printf("----------------------------------------------\n");
    printf("GPU Time:   %f ms\n", msec_total);
    printf("CPU Result: %f\n", cpu_result);
    printf("GPU Result: %f\n", gpu_result);
    printf("Difference: %e\n", diff);

    if (diff < 1e-2) { // Tolerance higher due to float vs double accumulation
        printf("Test PASSED\n");
    } else {
        printf("Test FAILED\n");
    }

    // 9. Cleanup
    CHECK_CUDA(cudaFreeHost(h_A));
    CHECK_CUDA(cudaFreeHost(h_B));
    CHECK_CUDA(cudaFreeHost(h_Partial));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_Partial));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}