#include "stdio.h"
#include <iostream>
#include <fstream>
// static void HandleError(cudaError_t err,
//                         const char *file,
//                         int line)
//                         {
//                             if(err != cudaSuccess)
//                             {
//                                 printf("%s in %s at line %d\n",
//                                 cudaGetErrorString(err),
//                                 file, line);
//                                 exit(EXIT_FAILURE);
//                             }
//                         }
// #define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

// int getThreadNum()
// {
//     cudaDeviceProp prop;
//     int count;

//     HANDLE_ERROR(cudaGetDeviceCount(&count));
//     printf("gpu num %d\n", count);
//     HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));
//     printf("max thread num: %d\n", prop.maxThreadsPerBlock);
//     printf("max grid dimensions: %d, %d, %d)\n",
//      prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
//     return prop.maxThreadsPerBlock;
// }

// __global__ void conv(float *img, float *kernel, float *result, 
//     int width, int height, int kernelSize)
//     {
//         int ti = threadIdx.x;
//         int bi = blockIdx.x;
//         int id = (bi * blockDim.x + ti);
//         if(id >= width * height)
//         {
//             return;
//         }
//         int row = id / width;
//         int col = id % width;
//         for(int i = 0; i < kernelSize; ++i)
//         {
//             for(int j = 0; j < kernelSize; ++j)
//             {
//                 float imgValue = 0;
//                 int curRow = row - kernelSize / 2 + i;
//                 int curCol = col - kernelSize / 2 + j;
//                 if(curRow < 0 || curCol < 0 || curRow >= height || curCol >= width)
//                 {}
//                 else
//                 {
//                     imgValue = img[curRow * width + curCol];
//                 }
//                 result[id] += kernel[i * kernelSize + j] * imgValue;
//             }

//         }
//     }

int main()
{
    
    int width = 25;
    int height = 25;
    int in_channels = 4;
    int out_channels = 4;
    int filter_width = 3;
    int filter_height = 3;
    int r = 1;

    float *input = new float[width * height * in_channels];
    float *filter = new float[filter_width * filter_height * in_channels * out_channels];
    float *bias = new float[in_channels];
    float *output = new float[width * height * out_channels];

    for(int in_channel = 0; in_channel < in_channels; in_channel++)
    {
        for(int row = 0; row < height; ++row)
        {
            for(int col = 0; col < width; ++col)
            {
                input[in_channel*width*height + row*width + col] = (col + row) % 50;
            }
        }
    }
    
    for(int in_channel = 0; in_channel < in_channels; in_channel++)
    {
        for(int out_channel = 0; out_channel < out_channels; out_channel++)
        {
            for(int h = 0; h < filter_height; h++)
            {
                for(int w = 0; w < filter_width; w++)
                {
                    if(h == r && w == r) filter[in_channel*out_channels*filter_height*filter_width + 
                                                out_channel*filter_height*filter_width + w*filter_width + h] = 1;
                    else filter[in_channel*out_channels*filter_height*filter_width + 
                                                out_channel*filter_height*filter_width + w*filter_width + h] = 0;
                }
            }
        }
    }

    for(int in_channel = 0; in_channel < in_channels; in_channel++)
        bias[in_channel] = 0;
    
    for(int out_channel = 0; out_channel < out_channels; out_channel++)
    {
        for(int row = 0; row < height; ++row)
        {
            for(int col = 0; col < width; ++col)
            {
                output[out_channel*width*height + row*width + col] = 0;
            }
        }
    }

    std::ofstream input_file("imput.txt");
    if(input_file.is_open())
    {
        for(int channel = 0; channel < in_channels; channel++)
        {
            for(int row = 0; row < height; row++)
            {
                for(int col = 0; col < width; col++)
                {
                    input_file << input[channel*width*height + row*width + col] << " ";
                }
                input_file << "\n";
            }
            input_file << "\n";
        }
        input_file.close();
    }

    std::ofstream filter_file("filter.txt");
    if(filter_file.is_open())
    {
        for(int in_channel = 0; in_channel < in_channels; in_channel++)
        {
            for(int out_channel = 0; out_channel < out_channels; out_channel++)
            {
                for(int h = 0; h < filter_height; h++)
                {
                    for(int w = 0; w < filter_width; w++)
                    {
                        filter_file << filter[in_channel*out_channels*filter_height*filter_width + 
                                            out_channel*filter_height*filter_width + w*filter_width + h] << " ";
                    }
                    filter_file << "\n";
                }
                filter_file << "\n";
            }
            filter_file << "\n";
        }
        input_file.close();
    }
    
    std::ofstream bias_file("bias.txt");
    if(bias_file.is_open())
    {
        for(int in_channel = 0; in_channel < in_channels; in_channel++)
        {
            bias_file << bias[in_channel] << " ";
        }
        bias_file.close();
    }

    std::ofstream output_file("output.txt");
    if(output_file.is_open())
    {
        for(int out_channel = 0; out_channel < out_channels; out_channel++)
        {
            for(int row = 0; row < height; row++)
            {
                for(int col = 0; col < width; col++)
                {
                    output_file << output[out_channel*width*height + row*width + col] << " ";
                }
                output_file << "\n";
            }
            output_file << "\n";
        }
        output_file.close();
    }

    // int kernelSize = 3;
    // float *kernel = new float[kernelSize * kernelSize];
    // for(int i = 0; i < kernelSize * kernelSize; ++i)
    // {
    //     kernel[i] = i % kernelSize - 1;
    // }

    // float *imgGpu;
    // float *kernelGpu;
    // float *resultGpu;

    // HANDLE_ERROR(cudaMalloc((void**)&imgGpu, width * height * sizeof(float)));
    // HANDLE_ERROR(cudaMalloc((void**)&kernelGpu, kernelSize * kernelSize * sizeof(float)));
    // HANDLE_ERROR(cudaMalloc((void**)&resultGpu, width * height * sizeof(float)));

    // HANDLE_ERROR(cudaMemcpy(imgGpu, img,
    //  width * height * sizeof(float), cudaMemcpyHostToDevice));
    // HANDLE_ERROR(cudaMemcpy(kernelGpu, kernel,
    //  kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice));
    
    // int threadNum = getThreadNum();
    // int blockNum = (width * height - 0.5) / threadNum + 1;

    // conv<<<blockNum, threadNum>> >
    //     (imgGpu, kernelGpu, resultGpu, width, height, kernelSize);

    // float *result = new float[width * height];
    // HANDLE_ERROR(cudaMemcpy(result, resultGpu,
    //  width * height * sizeof(float), cudaMemcpyDeviceToHost));

    // // visualization
    // printf("img\n");
    // for(int row = 0; row < 10; ++row)
    // {
    //     for(int col = 0; col < 10; ++col)
    //     {
    //         printf("%2.0f ", img[col + row * width]);
    //     }
    //     printf("\n");
    // }
    // printf("kernel\n");
    // for(int row = 0; row < kernelSize; ++row)
    // {
    //     for(int col = 0; col < kernelSize; ++col)
    //     {
    //         printf("%2.0f ", kernel[col + row * kernelSize]);
    //     }
    //     printf("\n");
    // }

    // printf("result\n");
    // for(int row = 0; row < 10; ++row)
    // {
    //     for(int col = 0; col < 10; ++col)
    //     {
    //         printf("%2.0f ", result[col + row * width]);
    //     }
    //     printf("\n");
    // }

    delete[] input;
    delete[] output;
    delete[] bias;
    delete[] filter;
    return 0;
}