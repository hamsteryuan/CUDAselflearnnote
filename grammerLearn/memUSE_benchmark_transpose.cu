#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_SIZE 32
#define MATRIX_SIZE 2048
//共享内存优势体现：注意这里举了一个转置矩阵的例子———因为矩阵内部不连续，对于读取速度要求比起数组加法更高
//结果提升了超过4倍！


// 简单版本：直接从全局内存读写（会有大量的非合并访问）
__global__ void transposeSimple(float* input, float* output, int n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < n && y < n) {
        // 读是合并的，但写是跨步的（性能差）
        output[x * n + y] = input[y * n + x];
    }
}

// 共享内存版本：使用tile来优化内存访问模式
__global__ void transposeSharedMem(float* input, float* output, int n) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1]; // +1 避免bank conflict
    
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    // 从全局内存读到共享内存（合并访问）
    if (x < n && y < n) {
        tile[threadIdx.y][threadIdx.x] = input[y * n + x];
    }
    
    __syncthreads();
    
    // 计算转置后的位置
    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;
    
    // 从共享内存写到全局内存（合并访问）
    if (x < n && y < n) {
        output[y * n + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// 朴素版本：最直接但最慢的实现
__global__ void transposeNaive(float* input, float* output, int n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < n && y < n) {
        // 读写都可能是非合并的
        output[x * n + y] = input[y * n + x];
    }
}

#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

void verifyTranspose(float* h_input, float* h_output, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (h_output[i * n + j] != h_input[j * n + i]) {
                printf("Error at (%d, %d): expected %.2f, got %.2f\n", 
                       i, j, h_input[j * n + i], h_output[i * n + j]);
                return;
            }
        }
    }
    printf("Verification PASSED!\n");
}

int main() {
    int n = MATRIX_SIZE;
    size_t bytes = n * n * sizeof(float);
    
    // Host memory
    float *h_input = (float*)malloc(bytes);
    float *h_output = (float*)malloc(bytes);
    
    // Initialize input matrix
    for (int i = 0; i < n * n; i++) {
        h_input[i] = (float)i;
    }
    
    // Device memory
    float *d_input, *d_output;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, bytes));
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    
    // CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    
    // Kernel configuration
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);
    
    printf("=== Matrix Transpose Performance Benchmark ===\n");
    printf("Matrix Size: %d x %d (%d elements)\n", n, n, n * n);
    printf("Memory Size: %.2f MB\n", bytes / (1024.0 * 1024.0));
    printf("Block Size: %d x %d\n", TILE_SIZE, TILE_SIZE);
    printf("Grid Size: %d x %d\n\n", gridSize.x, gridSize.y);
    
    const int NUM_ITERATIONS = 100;
    float naiveTime, simpleTime, sharedTime;
    
    // Warm-up
    printf("Running warm-up...\n");
    transposeNaive<<<gridSize, blockSize>>>(d_input, d_output, n);
    transposeSimple<<<gridSize, blockSize>>>(d_input, d_output, n);
    transposeSharedMem<<<gridSize, blockSize>>>(d_input, d_output, n);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Benchmark Naive version
    printf("Benchmarking Naive version...\n");
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        transposeNaive<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&naiveTime, start, stop));
    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));
    printf("  "); verifyTranspose(h_input, h_output, n);
    
    // Benchmark Simple version
    printf("Benchmarking Simple version...\n");
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        transposeSimple<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&simpleTime, start, stop));
    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));
    printf("  "); verifyTranspose(h_input, h_output, n);
    
    // Benchmark Shared Memory version
    printf("Benchmarking Shared Memory version...\n");
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        transposeSharedMem<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&sharedTime, start, stop));
    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));
    printf("  "); verifyTranspose(h_input, h_output, n);
    
    // Display results
    printf("\n=== Performance Results ===\n");
    printf("Iterations: %d\n\n", NUM_ITERATIONS);
    
    float bandwidth_naive = (2.0 * bytes * NUM_ITERATIONS / (1024.0*1024.0*1024.0)) / (naiveTime / 1000.0);
    float bandwidth_simple = (2.0 * bytes * NUM_ITERATIONS / (1024.0*1024.0*1024.0)) / (simpleTime / 1000.0);
    float bandwidth_shared = (2.0 * bytes * NUM_ITERATIONS / (1024.0*1024.0*1024.0)) / (sharedTime / 1000.0);
    
    printf("Naive Version:\n");
    printf("  Total: %.2f ms | Avg: %.4f ms | Bandwidth: %.2f GB/s\n", 
           naiveTime, naiveTime/NUM_ITERATIONS, bandwidth_naive);
    
    printf("\nSimple Version:\n");
    printf("  Total: %.2f ms | Avg: %.4f ms | Bandwidth: %.2f GB/s\n", 
           simpleTime, simpleTime/NUM_ITERATIONS, bandwidth_simple);
    
    printf("\nShared Memory Version:\n");
    printf("  Total: %.2f ms | Avg: %.4f ms | Bandwidth: %.2f GB/s\n", 
           sharedTime, sharedTime/NUM_ITERATIONS, bandwidth_shared);
    
    printf("\n=== Speedup Analysis ===\n");
    printf("Shared Memory vs Naive:  %.2fx faster (%.1f%% improvement)\n", 
           naiveTime/sharedTime, ((naiveTime-sharedTime)/naiveTime)*100);
    printf("Shared Memory vs Simple: %.2fx faster (%.1f%% improvement)\n", 
           simpleTime/sharedTime, ((simpleTime-sharedTime)/simpleTime)*100);
    
    printf("\n=== 为什么矩阵转置能体现共享内存优势？ ===\n");
    printf("1. 内存访问模式优化：\n");
    printf("   - Naive/Simple: 读或写有非合并访问(strided access)\n");
    printf("   - Shared Memory: 通过tile重组，读写都是合并访问\n\n");
    printf("2. Bank Conflict优化：\n");
    printf("   - 使用 [TILE_SIZE][TILE_SIZE+1] 避免共享内存bank冲突\n\n");
    printf("3. 数据重用：\n");
    printf("   - 每个tile被多次访问，充分利用共享内存的低延迟\n");
    
    // Cleanup
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    free(h_input);
    free(h_output);
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    
    printf("\nBenchmark completed!\n");
    return 0;
}
