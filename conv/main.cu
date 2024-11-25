#include <cudnn.h>
#include <iostream>
#include <opencv2/opencv.hpp>


#include 
#define checkCUDNN(expression)                                  \
  {                                                             \
    cudnnStatus_t status = (expression);                        \
    if (status != CUDNN_STATUS_SUCCESS) {                       \
      std::cerr << "Error on line " << __LINE__ << ": "       \
      << cudnnGetErrorString(status) << std::endl;            \
      std::exit(EXIT_FAILURE);                                \
    }                                                           \
 }



cv::Mat load_image(const char* image_path) {
  cv::Mat image = cv::imread(image_path, CV_LOAD_IMAGE_COLOR);
  image.convertTo(image, CV_32FC3);
  cv::normalize(image, image, 0, 1, cv::NORM_MINMAX);
  return image;
}

cv::Mat image = load_image("/path/to/image.png");

int main(int argc, const char* argv[]) {
  if (argc < 2) {
    std::cerr << "usage: conv <image> [gpu=0] [sigmoid=0]" << std::endl;
    std::exit(EXIT_FAILURE);
  }
 
  int gpu_id = (argc > 2) ? std::atoi(argv[2]) : 0;
  std::cerr << "GPU: " << gpu_id << std::endl;
 
  bool with_sigmoid = (argc > 3) ? std::atoi(argv[3]) : 0;
  std::cerr << "With sigmoid: " << std::boolalpha << with_sigmoid << std::endl;
 
  cv::Mat image = load_image(argv[1]);
 
  cudaSetDevice(gpu_id);
 
  cudnnHandle_t cudnn;
  cudnnCreate(&cudnn);
 
  // 输入张量的描述
  cudnnTensorDescriptor_t input_descriptor;
  checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
    /*format=*/CUDNN_TENSOR_NHWC, // 注意是 NHWC，TensorFlow更喜欢以 NHWC 格式存储张量(通道是变化最频繁的地方，即 BGR)，而其他一些更喜欢将通道放在前面
    /*dataType=*/CUDNN_DATA_FLOAT,
    /*batch_size=*/1,
    /*channels=*/3,
    /*image_height=*/image.rows,
    /*image_width=*/image.cols));
 
  // 卷积核的描述（形状、格式）
  cudnnFilterDescriptor_t kernel_descriptor;
  checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
  checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
    /*dataType=*/CUDNN_DATA_FLOAT,
    /*format=*/CUDNN_TENSOR_NCHW, // 注意是 NCHW
    /*out_channels=*/3,
    /*in_channels=*/3,
    /*kernel_height=*/3,
    /*kernel_width=*/3));
 
  // 卷积操作的描述（步长、填充等等）
  cudnnConvolutionDescriptor_t convolution_descriptor;
  checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
  checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
    /*pad_height=*/1,
    /*pad_width=*/1,
    /*vertical_stride=*/1,
    /*horizontal_stride=*/1,
    /*dilation_height=*/1,
    /*dilation_width=*/1,
    /*mode=*/CUDNN_CROSS_CORRELATION, // CUDNN_CONVOLUTION
    /*computeType=*/CUDNN_DATA_FLOAT));
 
  // 计算卷积后图像的维数
  int batch_size{ 0 }, channels{ 0 }, height{ 0 }, width{ 0 };
  checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convolution_descriptor,
    input_descriptor,
    kernel_descriptor,
    &batch_size,
    &channels,
    &height,
    &width));
 
  std::cerr << "Output Image: " << height << " x " << width << " x " << channels
    << std::endl;
 
  // 卷积输出张量的描述
  cudnnTensorDescriptor_t output_descriptor;
  checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
    /*format=*/CUDNN_TENSOR_NHWC,
    /*dataType=*/CUDNN_DATA_FLOAT,
    /*batch_size=*/1,
    /*channels=*/3,
    /*image_height=*/image.rows,
    /*image_width=*/image.cols));
 
  // 卷积算法的描述
  // cudnn_tion_fwd_algo_gemm——将卷积建模为显式矩阵乘法，
  // cudnn_tion_fwd_algo_fft——它使用快速傅立叶变换(FFT)进行卷积或
  // cudnn_tion_fwd_algo_winograd——它使用Winograd算法执行卷积。
  cudnnConvolutionFwdAlgo_t convolution_algorithm;
  checkCUDNN(
    cudnnGetConvolutionForwardAlgorithm(cudnn,
    input_descriptor,
    kernel_descriptor,
    convolution_descriptor,
    output_descriptor,
    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, // CUDNN_CONVOLUTION_FWD_SPECIFY_​WORKSPACE_LIMIT（在内存受限的情况下，memoryLimitInBytes 设置非 0 值）
    /*memoryLimitInBytes=*/0,
    &convolution_algorithm));
 
  // 计算 cuDNN 它的操作需要多少内存
  size_t workspace_bytes{ 0 };
  checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
    input_descriptor,
    kernel_descriptor,
    convolution_descriptor,
    output_descriptor,
    convolution_algorithm,
    &workspace_bytes));
  std::cerr << "Workspace size: " << (workspace_bytes / 1048576.0) << "MB"
    << std::endl;
  assert(workspace_bytes > 0);
 
  // *************************************************************************
  // 分配内存， 从 cudnnGetConvolutionForwardWorkspaceSize 计算而得
  void* d_workspace{ nullptr };
  cudaMalloc(&d_workspace, workspace_bytes);
 
  // 从 cudnnGetConvolution2dForwardOutputDim 计算而得
  int image_bytes = batch_size * channels * height * width * sizeof(float);
 
  float* d_input{ nullptr };
  cudaMalloc(&d_input, image_bytes);
  cudaMemcpy(d_input, image.ptr<float>(0), image_bytes, cudaMemcpyHostToDevice);
 
  float* d_output{ nullptr };
  cudaMalloc(&d_output, image_bytes);
  cudaMemset(d_output, 0, image_bytes);
  // *************************************************************************
  // clang-format off
  const float kernel_template[3][3] = {
    { 1, 1, 1 },
    { 1, -8, 1 },
    { 1, 1, 1 }
  };
  // clang-format on
 
  float h_kernel[3][3][3][3]; // NCHW
  for (int kernel = 0; kernel < 3; ++kernel) {
    for (int channel = 0; channel < 3; ++channel) {
      for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
          h_kernel[kernel][channel][row][column] = kernel_template[row][column];
        }
      }
    }
  }
 
  float* d_kernel{ nullptr };
  cudaMalloc(&d_kernel, sizeof(h_kernel));
  cudaMemcpy(d_kernel, h_kernel, sizeof(h_kernel), cudaMemcpyHostToDevice);
  // *************************************************************************
 
  const float alpha = 1.0f, beta = 0.0f;
 
  // 真正的卷积操作 ！！！前向卷积
  checkCUDNN(cudnnConvolutionForward(cudnn,
    &alpha,
    input_descriptor,
    d_input,
    kernel_descriptor,
    d_kernel,
    convolution_descriptor,
    convolution_algorithm,
    d_workspace, // 注意，如果我们选择不需要额外内存的卷积算法，d_workspace可以为nullptr。
    workspace_bytes,
    &beta,
    output_descriptor,
    d_output));
 
  if (with_sigmoid) {
    
    // 描述激活
    cudnnActivationDescriptor_t activation_descriptor;
    checkCUDNN(cudnnCreateActivationDescriptor(&activation_descriptor));
    checkCUDNN(cudnnSetActivationDescriptor(activation_descriptor,
      CUDNN_ACTIVATION_SIGMOID,
      CUDNN_PROPAGATE_NAN,
      /*relu_coef=*/0));
 
    // 前向 sigmoid 激活函数
    checkCUDNN(cudnnActivationForward(cudnn,
      activation_descriptor,
      &alpha,
      output_descriptor,
      d_output,
      &beta,
      output_descriptor,
      d_output));
    cudnnDestroyActivationDescriptor(activation_descriptor);
  }
 
  float* h_output = new float[image_bytes];
  cudaMemcpy(h_output, d_output, image_bytes, cudaMemcpyDeviceToHost);
 
  save_image("../cudnn-out.png", h_output, height, width);
 
  delete[] h_output;
  cudaFree(d_kernel);
  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_workspace);
 
  // 销毁
  cudnnDestroyTensorDescriptor(input_descriptor);
  cudnnDestroyTensorDescriptor(output_descriptor);
  cudnnDestroyFilterDescriptor(kernel_descriptor);
  cudnnDestroyConvolutionDescriptor(convolution_descriptor);
 
  cudnnDestroy(cudnn);
}