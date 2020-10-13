#include <cuda.h>
#include <opencv2/opencv.hpp>
#include <iostream>

#include "box_filter.h"
#include "math_kernels.h"

__device__ void compute_cov_var(float4 *mean_Ip, float4 *mean_II, float4 *mean_I,
        float4 *mean_p, float4 *var_I, float4 *cov_Ip, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int idx = y * width + x; 
    float4 m_I = mean_I[idx];
    var_I[idx] = mean_II[idx] - m_I * m_I;
    cov_Ip[idx] = mean_Ip[idx] - m_I * mean_p[idx];
}

__device__ void compute_ab(float4 *var_I, float4 *cov_Ip, float4 *mean_I,
        float4 *mean_p, float4 *a, float4 *b, float eps, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int idx = y * width + x; 
    float4 a_ = cov_Ip[idx] / (var_I[idx] + eps);
    a[idx] = a_;
    b[idx] = mean_p[idx] - a_ * mean_I[idx];
}

__device__ void compute_q(float4 *in, float4 *mean_a, float4 *mean_b, float4 *q,
        int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int idx = y * width + x; 
    float4 im_ = in[idx];
    q[idx] = mean_a[idx] * im_ + mean_b[idx];
}

__global__ void mean_kernel(float4* d_input,
        float4 *d_p,
        float4 *mean_I,
        float4 *mean_p,
        float4 *mean_Ip,
        float4 *mean_II,
        float4 *d_tmp,
        float4 *d_tmp2,
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

__global__ void cov_var_ab_kernel(float4* d_input,
        float4 *mean_I,
        float4 *mean_p,
        float4 *mean_Ip,
        float4 *mean_II,
        float4 *var_I,
        float4 *cov_Ip,
        float4 *a, 
        float4 *b,
        float4 *mean_a,
        float4 *mean_b,
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
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        box_filter(a, mean_a, width, height);
        box_filter(b, mean_b, width, height);
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

    int GRID_W = ceil(width / (float)TILE_W);
    int GRID_H = ceil(height / (float)TILE_H);

    const dim3 block(TILE_W, TILE_H);
    const dim3 grid(GRID_W, GRID_H);

    printf("grid_w: %d\n", grid.x);
    printf("grid_h: %d\n", grid.y);
    printf("block_w: %d\n", block.x);
    printf("block_h: %d\n", block.y);

    mean_kernel<<<grid, block>>>(d_input, d_p, d_mean_I, d_mean_p, d_mean_Ip,
            d_mean_II, d_tmp, d_tmp2, width, height, eps);

    cudaDeviceSynchronize();

    cov_var_ab_kernel<<<grid, block>>>(d_input, d_mean_I, d_mean_p, d_mean_Ip,
        d_mean_II, d_var_I, d_cov_Ip, d_a, d_b, d_mean_a, d_mean_b, 
        width, height, eps);

    cudaDeviceSynchronize();

    output_kernel<<<grid, block>>>(d_input, d_p, d_output, d_a, d_b,
            d_mean_a, d_mean_b, width, height, eps);

    cudaDeviceSynchronize();

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
