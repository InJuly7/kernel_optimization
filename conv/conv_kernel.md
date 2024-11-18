# Conv
==本目录写一些卷积算子的优化思想==

> 参考资料 Programming Massively Parallel Processors 

项目实现逻辑
- 首先为每一种conv写一个kernel
- 每一种kernel的策略 他们的参数配置,数据传输量 不同, 为每一个conv kernl写一个入口
- main函数通过 命令行配置 调用各个kernel的实现


首先我会根据 SeAFusion Yolov5s 中的 conv算子其中的shape 实现kernel优化
卷积实现算法
- 基本算法
- 通过矩阵乘法实现

# SeAFusion conv配置

- group = 1
- kernel_shape = [3,3]
- pads = [1,1,1,1]
- strides = [1,1]
- weight = [4,4,3,3]






# 基本算法
