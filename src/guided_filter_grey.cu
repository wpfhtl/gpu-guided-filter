#include <cuda.h>
#include <opencv2/opencv.hpp>
#include <iostream>

#include "box_filter_grey.h"

__device__ void compute_cov_var(float *mean_Ip, float *mean_II, float *mean_I,
        float *mean_p, float *var_I, float *cov_Ip, int width, int height)
{
    int x = blockIdx.x * TILE_W + threadIdx.x - RADIUS;
    int y = blockIdx.y * TILE_H + threadIdx.y - RADIUS;

    int idx = y * width + x; 
    float m_I = mean_I[idx];
    var_I[idx] = fmaxf(mean_II[idx] - m_I * m_I, 0.);
    cov_Ip[idx] = fmaxf(mean_Ip[idx] - m_I * mean_p[idx], 0.);
}

__device__ void compute_ab(float *var_I, float *cov_Ip, float *mean_I,
        float *mean_p, float *a, float *b, float eps, int width, int height)
{
    int x = blockIdx.x * TILE_W + threadIdx.x - RADIUS;
    int y = blockIdx.y * TILE_H + threadIdx.y - RADIUS;

    int idx = y * width + x; 
    float a_ = cov_Ip[idx] / (var_I[idx] + eps);
    a[idx] = a_;
    b[idx] = mean_p[idx] - a_ * mean_I[idx];
}

__device__ void compute_q(float *in, float *mean_a, float *mean_b, float *q,
        int width, int height)
{
    int x = blockIdx.x * TILE_W + threadIdx.x - RADIUS;
    int y = blockIdx.y * TILE_H + threadIdx.y - RADIUS;

    int idx = y * width + x; 
    float im_ = in[idx];
    q[idx] = mean_a[idx] * im_ + mean_b[idx];
}

__global__ void mean_kernel(float* d_input,
        float *d_p,
        float *d_q,
        float *mean_I,
        float *mean_p,
        float *mean_Ip,
        float *mean_II,
        float *var_I,
        float *cov_Ip,
        float *a,
        float *b,
        float *d_tmp,
        float *d_tmp2,
        float *mean_a,
        float *mean_b,
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

void guided_filter_cuda(float *h_input,
        float *h_p,
        float *h_output,
        float *h_tmp,
        float *h_tmp2,
        int width, int height,
        float eps)
{

    const int n = width * height * sizeof(float);

    float *d_input, *d_p, *d_output, *d_mean_I, *d_mean_p, *d_mean_Ip,
           *d_mean_II, *d_var_I, *d_cov_Ip, *d_a, *d_b, *d_mean_a,
           *d_mean_b, *d_tmp, *d_tmp2;

    checkCudaErrors(cudaMalloc<float>(&d_input, n));
    checkCudaErrors(cudaMalloc<float>(&d_p, n));
    checkCudaErrors(cudaMalloc<float>(&d_output, n));
    checkCudaErrors(cudaMalloc<float>(&d_mean_I, n));
    checkCudaErrors(cudaMalloc<float>(&d_mean_p, n));
    checkCudaErrors(cudaMalloc<float>(&d_mean_Ip, n));
    checkCudaErrors(cudaMalloc<float>(&d_mean_II, n));
    checkCudaErrors(cudaMalloc<float>(&d_var_I, n));
    checkCudaErrors(cudaMalloc<float>(&d_cov_Ip, n));
    checkCudaErrors(cudaMalloc<float>(&d_a, n));
    checkCudaErrors(cudaMalloc<float>(&d_b, n));
    checkCudaErrors(cudaMalloc<float>(&d_mean_a, n));
    checkCudaErrors(cudaMalloc<float>(&d_mean_b, n));
    checkCudaErrors(cudaMalloc<float>(&d_tmp, n));
    checkCudaErrors(cudaMalloc<float>(&d_tmp2, n));

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
    cv::Mat input = cv::imread(input_file);
    if(input.empty()) {
        std::cout<<"Input image Not Found: "<< input_file << std::endl;
        return;
    }
    cv::Mat g_original = cv::imread(g_file);
    if(g_original.empty()) {
        std::cout<<"guidance image Not Found: "<< g_file << std::endl;
        return;
    }

    cv::Mat g;
    g_original.convertTo(g, CV_32FC1);
    g /= 255.f;
    cv::imshow("guidance image", g);
    // cv::Mat p = inputRGBA.clone();
    cv::Mat p;
    input.convertTo(p, CV_32FC1);
    p /= 255.f;
    cv::imshow("input image", p);

    cv::Mat output (p.size(), g.type());

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

    cv::imshow("output image", output);
    cv::waitKey(0);
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
