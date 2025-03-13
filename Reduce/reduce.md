# 归约问题 N个元素 前N项和
## Reduce_v0_baseline.cu 解读
```cpp
#define THREAD_PER_BLOCK 256
const int N=32*1024*1024;
dim3 Grid( N/THREAD_PER_BLOCK,1);
dim3 Block( THREAD_PER_BLOCK,1);
reduce0<<<Grid,Block>>>(d_a,d_out);
```
grid_1D block_1D
每个线程块 256个元素, 一共 `N/256` 个线程块
那每个线程块都做了什么?? 每个线程又是如何执行的?? 归约问题又是被如何拆解的??

> N个元素归约, 拆分成N/256个子任务归约, 每个Block 执行一个子任务, 即256个元素求和,
> 首先 块内全部线程将输入数据Gmem搬运到Smem,块内同步,第一轮求和 只有一半的线程执行从Smem读数据 求和 覆盖Smem, 之后每轮求和元素减半, 最后线程号0将结果写回到输出数组

th_all : Gmem -> Smem
th_i : Smem[th_i] = Smem[th_i] + Smem[th_i+offset]
Gmem[Block_id] = Smem[th_0]

```cpp
__global__ void reduce0(float *d_in,float *d_out){
    __shared__ float sdata[THREAD_PER_BLOCK];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[tid] = d_in[i];
    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=1; s < blockDim.x; s *= 2) {
        if (tid % (2*s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) d_out[blockIdx.x] = sdata[0];
}
```


每个线程块 有256个线程 在smem上开辟256*4B大小的空间,容纳256个float元素
线程集合与输入数据集合关系: 每个线程对应输入数据的一个元素;
块内每个线程的全局索引对应的元素 读取到 smem
**主要关注线程的全局索引,与共享内存的映射**

一个线程块 执行256个元素求和 将结果写入共享内存的首地址 在搬运回全局内存输出数组中
归约思想是: 第一轮 只有128个线程活跃, 进行128次加法, 
代码中是 (0,1) (2,3) ... (i,j) ... (254,255)  线程号i,j 对应元素做加法, 写回共享内存(线程号i映射位置)
第二轮 只有 64个 线程活跃 (0,2) ... 
...
最后一轮时 s = blockDim.x/2 只有线程号0工作, `sdata[0] = sdata[0] + sdata[blockDim.x/2]`
最后只有线程号0 负责写回 只有一个线程活跃



## Reduce_v1_no_divergence_branch.cu 解读
```cpp
#define THREAD_PER_BLOCK 256
const int N=32*1024*1024;
dim3 Grid( N/THREAD_PER_BLOCK,1);
dim3 Block( THREAD_PER_BLOCK,1);
```
