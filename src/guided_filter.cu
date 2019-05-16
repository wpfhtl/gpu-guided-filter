#include <cuda.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>

#include "box_filter.h"

__device__ void compute_cov_var(float *mean_Ip, float *mean_II, float *mean_I,
        float *mean_p, float *var_I, float *cov_Ip,
        int width, int height)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = y * width + x; 
        float m_I = mean_I[idx];
        var_I[idx] = mean_Ip[idx] - m_I * m_I;
        cov_Ip[idx] = mean_II[idx] - m_I * mean_p[idx];
    }
}

__device__ void compute_ab(float *var_I, float *cov_Ip, float *mean_I,
        float *mean_p, float *a, float *b, float eps,
        int width, int height)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = y * width + x; 
        float a_ = cov_Ip[idx] / (var_I[idx] + eps);
        a[idx] = a_;
        b[idx] = mean_p[idx] - a_ * mean_I[idx];
    }
}

__device__ void compute_q(float *p, float *mean_a, float *mean_b, float *q,
        int width, int height)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = y * width + x; 
        float p_ = p[idx];
        q[idx] = mean_a[idx] * p_ + mean_b[idx];
    }
}

__global__ void guidedFilterCudaKernel(float* d_input,
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
        float *mean_a,
        float *mean_b,
        int width, int height,
        float eps)
{
    box_filter(d_input, mean_I, width, height);
    box_filter(d_input, mean_p, width, height);
    box_filter(d_input, mean_Ip, width, height);
    box_filter(d_input, mean_II, width, height);

    __syncthreads();

    compute_cov_var(mean_Ip, mean_II, mean_I, mean_p, var_I, cov_Ip, width, height);
    __syncthreads();
    compute_ab(var_I, cov_Ip, mean_I, mean_p, a, b, eps, width, height);
    __syncthreads();

    box_filter(a, mean_a, width, height);
    box_filter(b, mean_b, width, height);

    __syncthreads();

    compute_q(d_p, mean_a, mean_b, d_q, width, height);
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
        int width, int height,
        float eps)
{
    const int n = width * height * sizeof(float);
    float *d_input, *d_p, *d_output, *mean_I, *mean_p,* mean_Ip,
          *mean_II, *var_I, *cov_Ip, *a, *b, *mean_a, *mean_b;
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

    checkCudaErrors(cudaMemcpy(d_input, h_input, n, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_p, h_p, n, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_output, h_output, n, cudaMemcpyHostToDevice));

    int GRID_W = width / TILE_W + 1;
    int GRID_H = height / TILE_H + 1;

    const dim3 block(BLOCK_W, BLOCK_H);
    const dim3 grid(GRID_W, GRID_H);
    printf("grid_w: %d\n", grid.x);
    printf("grid_h: %d\n", grid.y);

    printf("block_w: %d\n", block.x);
    printf("block_h: %d\n", block.y);

    guidedFilterCudaKernel<<<grid,block>>>(d_input, d_p, d_output,
            mean_I, mean_p, mean_Ip, mean_II, var_I, cov_Ip, a, b,
            mean_a , mean_b, width, height, eps);

    cudaDeviceSynchronize();

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
}

void processUsingCuda(std::string input_file, std::string output_file) {
    cv::Mat input = cv::imread(input_file);
    if(input.empty()) {
        std::cout<<"Image Not Found: "<< input_file << std::endl;
        return;
    }

    cv::Mat inputGRAY;
    cvtColor(input, inputGRAY, CV_BGR2GRAY, 1);
    inputGRAY.convertTo(inputGRAY, CV_64F);
    inputGRAY /= 255;

    cv::Mat p = inputGRAY.clone();

    cv::Mat output (input.size(), inputGRAY.type());

    float eps = 0.2 * 0.2;

    guidedFilterCuda((float*) inputGRAY.ptr<float>(),
            (float*) p.ptr<float>(),
            (float*) output.ptr<float>(),
            inputGRAY.cols, inputGRAY.rows,
            eps);

    output *= 255;
    //output.convertTo(output, CV_32F);
    //cvtColor(output, output, CV_GRAY2BGR, 3);

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
