## cuda一些基本概念和事情

## GPU层级
SM vs 线程块 vs 线程：层级关系
GPU > SM > 线程块 > 线程（每级都是包含多个）
SM是硬件执行单元
每个SM都拥有自己独享的一小块、但速度极快的共享内存。不同SM之间完全隔离。典型大小如64KB(很珍贵)
线程块是调度单元
GPU调度器以线程块为单位将工作分配给SM，而不是单个线程。线程块之间通信隔离，即使可能基于同样的硬件基础。
同步边界：线程块内的线程可以通过__syncthreads()同步，但线程块之间无法同步
线程：最基础的执行单元，和CPU的线程相似。
为了达到高并发，应该：
优化寄存器使用
合理配置线程块大小（如128-256线程/块）
平衡共享内存的使用
## 中文输出
chcp 65001  文件用utf-8编码
* 中文标点符号似乎依旧乱码
一些中文注释会出乱码warning，不影响实际结果
# 编译后的lib与exp文件
```
nvcc StreamShow.cu -o StreamShow
tmpxft_00008550_00000000-10_StreamShow.cudafe1.cpp
  正在创建库 StreamShow.lib 和对象 StreamShow.exp
```
MSVC 链接器的默认行为（即使没有显式导出，也会生成导出文件）
暂时是没有用的，可以忽略

## kernel 函数
kernelFunction<<<gridDim, blockDim, sharedMem, stream>>>(arguments);
**这是异步函数：这行代码分配给了GPU要完成的任务，这行代码执行后，CPU立即继续往下走，不会等待GPU完成，需要同步**
kernelFunction  需要接 \__global__ 定义
1. Grid维度：指定启动多少个线程块（Block）
表示只启动1个Block
2.Block维度:指定每个Block中有多少个线程（Thread）
3. 共享内存大小:动态分配的共享内存大小
动态分配的共享内存是指在核函数启动时才确定大小的共享内存，而不是在编译时固定。
**从共享内存读取比全局内存快得多（GPU显存芯片整体使用，GB级别）**
这是每个线程块的固定分配大小，不是上限
4. stream: 指定在哪个CUDA流上执行用于异步执行，不同流可以同时执行
## 同步操作 
上文说了，kernel是异步执行，如果要同步（等一下GPU的执行完成），可以由以下方法
## Host---Device指代
主机（Host）

指的是 CPU 及其内存（RAM）
运行普通的 C/C++ 代码
h_data 中的 h 前缀通常表示 "host"

设备（Device）

指的是 GPU 及其显存（VRAM）
运行 CUDA 核函数（__global__ 函数）
d_data 中的 d 前缀通常表示 "device"

## Stream
Steam 是 CUDA 中的一个任务队列，用于管理在 GPU 上执行的操作序列。同一个流中的操作按顺序执行，但不同流之间可以并发执行。
CUDA会有一个默认隐式流