// StreamShow.cu
#include <iostream>
#include <cuda_runtime.h>
// 简单的内核函数：计算 1 + 1
__global__ void addKernel(int *result) {
    *result = 1 + 1;
}

int main() {
    int *d_result;
    int h_result;
    // 分配设备内存
    cudaMalloc(&d_result, sizeof(int));

    // 1. 默认流执行（同步）
    std::cout << "Default stream execution..." << std::endl;
    addKernel<<<1, 1>>>(d_result);  // 在默认流中启动内核
    //这里addKernel要执行的函数，也定义了此时要执行的kernel，可以当成一个行动对象
    //<<<1, 1>>> block 数量 和每个 block 线程数
    //（）表示传递的参数，这里传递的其实本质是一个会被设备用的内存位置
 
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);  
       // 把d_result 传递到&h_result的位置
       //（目的地，来源，大小，传输方向）
    std::cout << "Result: " << h_result << std::endl;

    // 2. 特定流执行（异步）
    cudaStream_t stream;
    cudaStreamCreate(&stream);  // 创建特定流
    std::cout << "Specific stream asynchronous execution..." << std::endl;
    addKernel<<<1, 1, 0, stream>>>(d_result);  // 在特定流中启动内核
    //这是最完整的核声明，这里0是代表没有额外的共享内存分配，这个也是其默认值，stream为特定流
    // 不等待，直接继续主机代码
    std::cout << "Host continues execution without waiting for GPU..." << std::endl;

    // 3. 流同步
    cudaStreamSynchronize(stream);  // 等待特定流完成
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "Asynchronous result: " << h_result << std::endl;

    // 清理
    cudaStreamDestroy(stream);
    cudaFree(d_result);
    return 0;
}