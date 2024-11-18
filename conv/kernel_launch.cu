#include "cuda.cuh"

#define RADIUS 1
void conv2D_basic_kernel(const float *input,const float *weights, float *output, float *bias, int width, 
                        int height, int in_channels, int out_channels)
{
    float* input_;
    float* weights_;
    float* bias_;
    float* output_;

    int kernel_shape = 2*RADIUS+1;
    cudaMalloc((void**)&input_, width*height*in_channels*sizeof(float));
    cudaMalloc((void**)&weights_, (2*RADIUS+1)*(2*RADIUS+1)*in_channels*out_channels*sizeof(float));
    cudaMalloc((void**)&bias_, out_channels*sizeof(float));
    cudaMalloc((void**)&output_, width*height*out_channels*sizeof(float));

    cudaMemcpy(input_,input,width*height*in_channels*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(weights_,weights,(2*RADIUS+1)*(2*RADIUS+1)*in_channels*out_channels*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(bias_, bias_, out_channels*sizeof(float),cudaMemcpyHostToDevice);

    conv2D_basic_
}