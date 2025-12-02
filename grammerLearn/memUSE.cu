#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 256
#define ARRAY_SIZE 1024

// CUDA kernel using shared memory for array addition
__global__ void addArraysSharedMem(float* a, float* b, float* c, int n) {
    // Shared memory allocation
    __shared__ float shared_a[BLOCK_SIZE];
    __shared__ float shared_b[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    if (idx < n) { （重要！！！一定加越界判断）
        shared_a[tid] = a[idx];
        shared_b[tid] = b[idx];
    } else {
        shared_a[tid] = 0.0f;
        shared_b[tid] = 0.0f;
    }
    
    // Synchronize threads in the block
    __syncthreads();
    
    // Perform computation using shared memory
    //使用：每个线程用共享内存中的数据做加法（快）
    if (idx < n) {
        c[idx] = shared_a[tid] + shared_b[tid];
    }
}

// Simple kernel without shared memory for comparison
__global__ void addArraysSimple(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Error checking macro Error处理方法
#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

int main() {
    // Host memory pointers
    float *h_a, *h_b, *h_c;
    // Device memory pointers
    float *d_a, *d_b, *d_c;
    
    size_t bytes = ARRAY_SIZE * sizeof(float);
    
    // 1. Allocate host memory
    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);
    h_c = (float*)malloc(bytes);
    
    // Initialize host arrays
    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
    }
    
    // 2. Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc(&d_a, bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_b, bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_c, bytes));
    
    // 3. Copy data from host to device
    printf("Copying data from host to device...\n");
    CHECK_CUDA_ERROR(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    
    // 4. Launch kernel with shared memory
    dim3 blockSize(BLOCK_SIZE);
    dim3 gridSize((ARRAY_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    printf("Launching kernel with shared memory...\n");
    addArraysSharedMem<<<gridSize, blockSize>>>(d_a, d_b, d_c, ARRAY_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // 5. Copy result back from device to host
    printf("Copying result from device to host...\n");
    CHECK_CUDA_ERROR(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    
    // 6. Verify results
    bool success = true;
    for (int i = 0; i < ARRAY_SIZE; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            printf("Error at index %d: %f != %f\n", i, h_c[i], h_a[i] + h_b[i]);
            success = false;
            break;
        }
    }
    
    if (success) {
        printf("Array addition successful!\n");
        printf("First 10 results: ");
        for (int i = 0; i < 10; i++) {
            printf("%.1f ", h_c[i]);
        }
        printf("\n");
    }
    
    // 7. Demonstrate different memory types
    printf("\n=== Memory Information ===\n");
    
    // Device memory info
    size_t free_mem, total_mem;
    CHECK_CUDA_ERROR(cudaMemGetInfo(&free_mem, &total_mem));
    printf("GPU Memory - Total: %.2f MB, Free: %.2f MB\n", 
           total_mem / 1024.0 / 1024.0, free_mem / 1024.0 / 1024.0);
    
    // Demonstrate unified memory (if supported)
    float *unified_ptr;
    if (cudaMallocManaged(&unified_ptr, bytes) == cudaSuccess) {
        printf("Unified memory allocation successful\n");
        cudaFree(unified_ptr);
    }
    
    // 8. Clean up memory
    free(h_a);
    free(h_b);
    free(h_c);
    
    CHECK_CUDA_ERROR(cudaFree(d_a));
    CHECK_CUDA_ERROR(cudaFree(d_b));
    CHECK_CUDA_ERROR(cudaFree(d_c));
    
    printf("Memory cleanup completed\n");
    
    return 0;
}