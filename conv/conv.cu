#define RADIUS 1
// kernel_shape [3,3]
// group 1
// pad [1,1,1,1]
// stride [1,1]
__global__ void conv2D_basic(float *input_, float *weights_, float *bias_, float *output_, int width, int height, int in_channels, int out_channels)
{
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    
    float out_value = 0;
    for(int out_channel; out_channel < out_channels; out_channel++)
    {
        for(int in_channel = 0; in_channel < in_channels; in_channel++)
        {
            for(int fRow = 0; fRow < 2*RADIUS+1; fRow++)
            {
                for(int fCol = 0; fCol < 2*RADIUS+1; fCol++)
                {
                    int inRow = outRow - RADIUS + fRow; 
                    int inCol = outCol - RADIUS + fCol;
                    if(inRow >= 0 && inRow < height && inCol >= 0 && inCol < width)
                        out_value += weights_[out_channel*in_channel*(2*RADIUS+1)*(2*RADIUS+1)+ in_channel*(2*RADIUS+1)*(2*RADIUS+1) 
                                            + inRow*(2*RADIUS+1) + inCol] * input_[in_channel*width*height + inRow+width + inCol];
                }
            }
        }
        output_[out_channel*outRow*outCol + outRow*height + outCol] = out_value + bias_[out_channel];
    }
}

