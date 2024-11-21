#!/bin/bash

# 检查是否传入参数
if [ "$#" -ne 1 ]; then
    echo "Usage: ./run.sh --sm_<compute_capability>"
    exit 1
fi

# 获取架构参数
if [[ "$1" =~ --sm_([0-9]+) ]]; then
    ARCH="sm_${BASH_REMATCH[1]}"
else
    echo "Invalid argument. Use format: --sm_<compute_capability> (e.g., --sm_61)"
    exit 1
fi

# 编译 CUDA 程序
nvcc -arch=${ARCH} -O2 main.cu -o main
if [ $? -ne 0 ]; then
    echo "Compilation failed."
    exit 1
fi
echo "compile success"

# 根据架构执行不同的命令
if [[ "$ARCH" == "sm_61" ]]; then
    ./main
    rm main
elif [[ "$ARCH" == "sm_75" ]]; then
    ncu -o kernel_profile -f --set full ./main
else
    echo "Unsupported architecture for additional actions."
fi
