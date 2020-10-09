#include <cuda.h>
#include <opencv2/opencv.hpp>
#include <iostream>

#include "box_filter.h"
#include "math_kernels.h"

__device__ void compute_cov_var(float4 *mean_Ip, float4 *mean_II, float4 *mean_I,
        float4 *mean_p, float4 *var_I, float4 *cov_Ip, int width, int height)
{
    int x = blockIdx.x * TILE_W + threadIdx.x - RADIUS;
    int y = blockIdx.y * TILE_H + threadIdx.y - RADIUS;

    int idx = y * width + x; 
    float4 m_I = mean_I[idx];
    var_I[idx] = fmaxf(mean_II[idx] - m_I * m_I, 0.);
    cov_Ip[idx] = fmaxf(mean_Ip[idx] - m_I * mean_p[idx], 0.);
}

__device__ void compute_ab(float4 *var_I, float4 *cov_Ip, float4 *mean_I,
        float4 *mean_p, float4 *a, float4 *b, float eps, int width, int height)
{
    int x = blockIdx.x * TILE_W + threadIdx.x - RADIUS;
    int y = blockIdx.y * TILE_H + threadIdx.y - RADIUS;

    int idx = y * width + x; 
    float4 a_ = cov_Ip[idx] / (var_I[idx] + eps);
    a[idx] = a_;
    b[idx] = mean_p[idx] - a_ * mean_I[idx];
}

__device__ void compute_q(float4 *in, float4 *mean_a, float4 *mean_b, float4 *q,
        int width, int height)
{
    int x = blockIdx.x * TILE_W + threadIdx.x - RADIUS;
    int y = blockIdx.y * TILE_H + threadIdx.y - RADIUS;

    int idx = y * width + x; 
    float4 im_ = in[idx];
    q[idx] = mean_a[idx] * im_ + mean_b[idx];
}

__global__ void mean_kernel(float4* d_input,
        float4 *d_p,
        float4 *d_q,
        float4 *mean_I,
        float4 *mean_p,
        float4 *mean_Ip,
        float4 *mean_II,
        float4 *var_I,
        float4 *cov_Ip,
        float4 *a,
        float4 *b,
        float4 *d_tmp,
        float4 *d_tmp2,
        float4 *mean_a,
        float4 *mean_b,
        int width, int height,
        float eps)
{
    int x = blockIdx.x * TILE_W + threadIdx.x - RADIUS;
    int y = blockIdx.y * TILE_H + threadIdx.y - RADIUS;
    box_filter(d_input, mean_I, width, height);
    box_filter(d_p, mean_p, width, height);
    __syncthreads();
    box_filter(d_tmp, mean_Ip, width, height);
    box_filter(d_tmp2, mean_II, width, height);
    if (x >= 0 && y >= 0 && x < width && y < height) {
    compute_cov_var(mean_Ip, mean_II, mean_I, mean_p, var_I, cov_Ip, width, height);
    compute_ab(var_I, cov_Ip, mean_I, mean_p, a, b, eps, width, height);
    }
}

__global__ void output_kernel(float4* d_input,
        float4 *d_p,
        float4 *d_q,
        float4 *a,
        float4 *b,
        float4 *mean_a,
        float4 *mean_b,
        int width, int height,
        float eps)
{
    int x = blockIdx.x * TILE_W + threadIdx.x - RADIUS;
    int y = blockIdx.y * TILE_H + threadIdx.y - RADIUS;

     box_filter(a, mean_a, width, height);
    __syncthreads();
    box_filter(b, mean_b, width, height);
    __syncthreads();

    if (x >= 0 && y >= 0 && x < width && y < height) {
    compute_q(d_p, mean_a, mean_b, d_q, width, height);
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

void guided_filter_cuda(float4 *h_input,
        float4 *h_p,
        float4 *h_output,
        float4 *h_tmp,
        float4 *h_tmp2,
        int width, int height,
        float eps)
{

    const int n = width * height * sizeof(float4);

    float4 *d_input, *d_p, *d_output, *d_mean_I, *d_mean_p, *d_mean_Ip,
           *d_mean_II, *d_var_I, *d_cov_Ip, *d_a, *d_b, *d_mean_a,
           *d_mean_b, *d_tmp, *d_tmp2;

    checkCudaErrors(cudaMalloc<float4>(&d_input, n));
    checkCudaErrors(cudaMalloc<float4>(&d_p, n));
    checkCudaErrors(cudaMalloc<float4>(&d_output, n));
    checkCudaErrors(cudaMalloc<float4>(&d_mean_I, n));
    checkCudaErrors(cudaMalloc<float4>(&d_mean_p, n));
    checkCudaErrors(cudaMalloc<float4>(&d_mean_Ip, n));
    checkCudaErrors(cudaMalloc<float4>(&d_mean_II, n));
    checkCudaErrors(cudaMalloc<float4>(&d_var_I, n));
    checkCudaErrors(cudaMalloc<float4>(&d_cov_Ip, n));
    checkCudaErrors(cudaMalloc<float4>(&d_a, n));
    checkCudaErrors(cudaMalloc<float4>(&d_b, n));
    checkCudaErrors(cudaMalloc<float4>(&d_mean_a, n));
    checkCudaErrors(cudaMalloc<float4>(&d_mean_b, n));
    checkCudaErrors(cudaMalloc<float4>(&d_tmp, n));
    checkCudaErrors(cudaMalloc<float4>(&d_tmp2, n));

    checkCudaErrors(cudaMemcpy(d_input, h_input, n, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_p, h_p, n, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_output, h_output, n, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_tmp, h_tmp, n, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_tmp2, h_tmp2, n, cudaMemcpyHostToDevice));

    int GRID_W = ceil(width /(float)TILE_W);
    int GRID_H = ceil(height / (float)TILE_H);

    const dim3 block(BLOCK_W, BLOCK_H);
    const dim3 grid(GRID_W, GRID_H);

    printf("grid_w: %d\n", grid.x);
    printf("grid_h: %d\n", grid.y);
    printf("block_w: %d\n", block.x);
    printf("block_h: %d\n", block.y);

    mean_kernel<<<grid, block>>>(d_input, d_p, d_output, d_mean_I, d_mean_p, d_mean_Ip,
            d_mean_II, d_var_I, d_cov_Ip, d_a, d_b, d_tmp, d_tmp2, d_mean_a,
            d_mean_b, width, height, eps);

    cudaDeviceSynchronize();
    output_kernel<<<grid, block>>>(d_input, d_p, d_output, d_a, d_b,
            d_mean_a, d_mean_b, width, height, eps);

    auto error = cudaGetLastError();
    if (error != cudaSuccess)
        printf("An error occured with CUDA: %s\n", cudaGetErrorString(error));

    checkCudaErrors(cudaMemcpy(h_output, d_output, n, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_input));
    checkCudaErrors(cudaFree(d_p));
    checkCudaErrors(cudaFree(d_output));
    checkCudaErrors(cudaFree(d_mean_I));
    checkCudaErrors(cudaFree(d_mean_p));
    checkCudaErrors(cudaFree(d_mean_Ip));
    checkCudaErrors(cudaFree(d_mean_II));
    checkCudaErrors(cudaFree(d_var_I));
    checkCudaErrors(cudaFree(d_cov_Ip));
    checkCudaErrors(cudaFree(d_a));
    checkCudaErrors(cudaFree(d_b));
    checkCudaErrors(cudaFree(d_mean_a));
    checkCudaErrors(cudaFree(d_mean_b));
    checkCudaErrors(cudaFree(d_tmp));
    checkCudaErrors(cudaFree(d_tmp2));
}

void compute(std::string input_file, std::string g_file, std::string output_file) {
    cv::Mat p = cv::imread(input_file);
    if(p.empty()) {
        std::cout<<"Input image Not Found: "<< input_file << std::endl;
        return;
    }
    cv::Mat g = cv::imread(g_file);
    if(g.empty()) {
        std::cout<<"guidance image Not Found: "<< g_file << std::endl;
        return;
    }

    g.convertTo(g, CV_32FC1);
    g /= 255.f;

    // cv::Mat p = inputRGBA.clone();
    p.convertTo(p, CV_32FC1);
    p /= 255.f;

    cv::Mat output (input.size(), g.type());

    float eps = 0.2 * 0.2;
    cv::Mat tmp = g.mul(p);
    cv::Mat tmp2 = g.mul(g);

    guided_filter_cuda(g.ptr<float>(),
            p.ptr<float>(),
            output.ptr<float>(),
            tmp.ptr<float>(),
            tmp2.ptr<float>(),
            g.cols, g.rows,
            eps);

    output *= 255;

    imwrite(output_file, output);
    printf("Saved image: %s\n", output_file);
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cout << "Choose image" << std::endl;
        return 1;
    }
    compute(argv[argc - 2], argv[argc - 1], "out.png");
    return 0;
}
