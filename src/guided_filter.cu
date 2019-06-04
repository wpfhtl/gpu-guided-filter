#include <cuda.h>
#include <opencv2/opencv.hpp>
#include <iostream>

#include "box_filter.h"
#include "math_kernels.h"

__device__ void compute_cov_var(float *mean_Ip, float *mean_II, float *mean_I,
        float *mean_p, float *var_I, float *cov_Ip,
        int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int idx = y * width + x; 
    float m_I = mean_I[idx];
    var_I[idx] = max(mean_II[idx] - m_I * m_I, 0.);
    cov_Ip[idx] = max(mean_Ip[idx] - m_I * mean_p[idx], 0.);
}

__device__ void compute_ab(float *var_I, float *cov_Ip, float *mean_I,
        float *mean_p, float *a, float *b, float eps,
        int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int idx = y * width + x; 
    float a_ = cov_Ip[idx] / (var_I[idx] + eps);
    a[idx] = a_;
    b[idx] = mean_p[idx] - a_ * mean_I[idx];
}

__device__ void compute_q(float *in, float *mean_a, float *mean_b, float *q,
        int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int idx = y * width + x; 
    float im_ = in[idx];
    q[idx] = mean_a[idx] * im_ + mean_b[idx];
}

__global__ void mean_kernel(float* d_input,
        float *d_p,
        float *mean_I,
        float *mean_p,
        float *mean_Ip,
        float *mean_II,
        float *d_tmp,
        float *d_tmp2,
        int width, int height,
        float eps)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        box_filter(d_input, mean_I, width, height);
        box_filter(d_p, mean_p, width, height);
        box_filter(d_tmp, mean_Ip, width, height);
        box_filter(d_tmp2, mean_II, width, height);
    }
}

__global__ void cov_var_ab_kernel(float* d_input,
        float *mean_I,
        float *mean_p,
        float *mean_Ip,
        float *mean_II,
        float *var_I,
        float *cov_Ip,
        float *a, 
        float *b,
        float *mean_a,
        float *mean_b,
        int width, int height,
        float eps)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        compute_cov_var(mean_Ip, mean_II, mean_I, mean_p, var_I, cov_Ip, width, height);
        compute_ab(var_I, cov_Ip, mean_I, mean_p, a, b, eps, width, height);
    }
}

__global__ void output_kernel(float* d_input,
        float *d_p,
        float *d_q,
        float *a, 
        float *b,
        float *mean_a,
        float *mean_b,
        int width, int height,
        float eps)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {

        box_filter(a, mean_a, width, height);
        box_filter(b, mean_b, width, height);
        compute_q(d_p, mean_a, mean_b, d_q, width, height);
        //d_q[y * width + x] = mean_a[y * width + x];
    }
}

#define checkCudaErrors(err)           __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError err, const char *file, const int line)
{
    if (cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
                file, line, (int)err, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void guidedFilterCuda(float *h_input,
        float *h_p,
        float *h_output,
        float *h_tmp,
        float *h_tmp2,
        int width, int height,
        float eps)
{

    const int n = width * height * sizeof(float);

    float *d_input, *d_p, *d_output, *mean_I, *mean_p,* mean_Ip,
          *mean_II, *var_I, *cov_Ip, *a, *b, *mean_a, *mean_b, *d_tmp, *d_tmp2;
    checkCudaErrors(cudaMalloc<float>(&d_input, n));
    checkCudaErrors(cudaMalloc<float>(&d_p, n));
    checkCudaErrors(cudaMalloc<float>(&d_output, n));
    checkCudaErrors(cudaMalloc<float>(&mean_I, n));
    checkCudaErrors(cudaMalloc<float>(&mean_p, n));
    checkCudaErrors(cudaMalloc<float>(&mean_Ip, n));
    checkCudaErrors(cudaMalloc<float>(&mean_II, n));
    checkCudaErrors(cudaMalloc<float>(&var_I, n));
    checkCudaErrors(cudaMalloc<float>(&cov_Ip, n));
    checkCudaErrors(cudaMalloc<float>(&a, n));
    checkCudaErrors(cudaMalloc<float>(&b, n));
    checkCudaErrors(cudaMalloc<float>(&mean_a, n));
    checkCudaErrors(cudaMalloc<float>(&mean_b, n));
    checkCudaErrors(cudaMalloc<float>(&d_tmp, n));
    checkCudaErrors(cudaMalloc<float>(&d_tmp2, n));

    checkCudaErrors(cudaMemcpy(d_input, h_input, n, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_p, h_p, n, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_output, h_output, n, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_tmp, h_tmp, n, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_tmp2, h_tmp2, n, cudaMemcpyHostToDevice));

    int GRID_W = ceil(width / (float)TILE_W);
    int GRID_H = ceil(height / (float)TILE_H);

    const dim3 block(TILE_W, TILE_H);
    const dim3 grid(GRID_W, GRID_H);
    //const dim3 grid(width/(block.x)+ block.x,height/(block.y)+block.y);
    //const dim3 grid((width + block.x-1)/block.x, (height + block.y - 1)/block.y);
    printf("grid_w: %d\n", grid.x);
    printf("grid_h: %d\n", grid.y);

    printf("block_w: %d\n", block.x);
    printf("block_h: %d\n", block.y);

    mean_kernel<<<grid, block>>>(d_input, d_p, mean_I, mean_p, mean_Ip,
            mean_II, d_tmp, d_tmp2, width, height, eps);

    cudaDeviceSynchronize();

    cov_var_ab_kernel<<<grid, block>>>(d_input, mean_I, mean_p, mean_Ip,
        mean_II, var_I, cov_Ip, a, b, mean_a, mean_b, width, height,
        eps);

    cudaDeviceSynchronize();

    output_kernel<<<grid, block>>>(d_input, d_p, d_output, a, b, mean_a, mean_b,
        width, height, eps);

    cudaDeviceSynchronize();

    auto error = cudaGetLastError();
    if (error != cudaSuccess)
        printf("An error occured with CUDA: %s\n", cudaGetErrorString(error));

    checkCudaErrors(cudaMemcpy(h_output, d_output, n, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_input));
    checkCudaErrors(cudaFree(d_p));
    checkCudaErrors(cudaFree(d_output));
    checkCudaErrors(cudaFree(mean_I));
    checkCudaErrors(cudaFree(mean_p));
    checkCudaErrors(cudaFree(mean_Ip));
    checkCudaErrors(cudaFree(mean_II));
    checkCudaErrors(cudaFree(var_I));
    checkCudaErrors(cudaFree(cov_Ip));
    checkCudaErrors(cudaFree(a));
    checkCudaErrors(cudaFree(b));
    checkCudaErrors(cudaFree(mean_a));
    checkCudaErrors(cudaFree(mean_b));
    checkCudaErrors(cudaFree(d_tmp));
    checkCudaErrors(cudaFree(d_tmp2));
}

void processUsingCuda(std::string input_file, std::string output_file) {
    cv::Mat input = cv::imread(input_file);
    if(input.empty()) {
        std::cout<<"Image Not Found: "<< input_file << std::endl;
        return;
    }

    cv::Mat inputGRAY;
    cvtColor(input, inputGRAY, CV_BGR2GRAY, 1);
    inputGRAY.convertTo(inputGRAY, CV_32F);
    inputGRAY /= 255.f;

    cv::Mat p = inputGRAY.clone();

    cv::Mat output (input.size(), inputGRAY.type());

    float eps = 0.2 * 0.2;
    cv::Mat tmp = inputGRAY.mul(p);
    cv::Mat tmp2 = inputGRAY.mul(inputGRAY);

    //std::cout << inputGRAY << std::endl;
    guidedFilterCuda(inputGRAY.ptr<float>(),
            p.ptr<float>(),
            output.ptr<float>(),
            tmp.ptr<float>(),
            tmp2.ptr<float>(),
            inputGRAY.cols, inputGRAY.rows,
            eps);

    output *= 255;

    //std::cout << output << std::endl;
    //output.convertTo(output, CV_32F);
    cvtColor(output, output, CV_GRAY2BGR, 3);

    imwrite(output_file, output);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cout << "Choose image" << std::endl;
        return 1;
    }
    processUsingCuda(argv[argc - 1], "out.png");
    return 0;
}
