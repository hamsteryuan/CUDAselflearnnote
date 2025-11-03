#include <stdio.h>
#include <cuda_runtime.h>

// Very small example: launch 10 threads, each computes 1+1 and prints the result.
__global__ void simpleAdd()
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Thread %d: 1 + 1 = %d\n", id, 1 + 1);
}

int main()
{
    printf("Hello from CPU\n");

    // Launch 1 block of 10 threads
    simpleAdd<<<1, 10>>>();

    // Wait for GPU to finish 
    cudaDeviceSynchronize();

    printf("Back to CPU\n");
    return 0;
}