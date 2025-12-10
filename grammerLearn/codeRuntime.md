# #include <cuda_runtime.h> 使用

## 使用函数

### 存储传递 
CPU RAM
GPU VRAM
```
// 在 GPU 上分配内存
float *d_data;
cudaMalloc(&d_data, 1024 * sizeof(float));

// 在主机和设备之间复制数据
float h_data[1024];
cudaMemcpy(d_data, h_data, 1024 * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(h_data, d_data, 1024 * sizeof(float), cudaMemcpyDeviceToHost);

// 释放 GPU 内存
cudaFree(d_data);
```

### 同步
// 等待所有 GPU 操作完成
cudaDeviceSynchronize(); 

// 等待特定流完成
cudaStreamSynchronize(stream);