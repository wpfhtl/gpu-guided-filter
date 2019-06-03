#include <cuda.h>
#include <opencv2/opencv.hpp>
#include <iostream>

#include "box_filter.h"
#include "math_kernels.h"

__device__ void compute_cov_var(float4 *mean_Ip, float4 *mean_II, float4 *mean_I,
        float4 *mean_p, float4 *var_I, float4 *cov_Ip,
        int width, int height)
{
    int x = blockIdx.x * TILE_W + threadIdx.x - RADIUS;
    int y = blockIdx.y * TILE_H + threadIdx.y - RADIUS;
    if (x >= 0 && y >= 0 && x < width && y < height) {
        int idx = y * width + x; 
        float4 m_I = mean_I[idx];
        var_I[idx] = mean_II[idx] - m_I * m_I;
        cov_Ip[idx] = mean_Ip[idx] - m_I * mean_p[idx];
    }
}

__device__ void compute_ab(float4 *var_I, float4 *cov_Ip, float4 *mean_I,
        float4 *mean_p, float4 *a, float4 *b, float eps,
        int width, int height)
{
    int x = blockIdx.x * TILE_W + threadIdx.x - RADIUS;
    int y = blockIdx.y * TILE_H + threadIdx.y - RADIUS;
    if (x >= 0 && y >= 0 && x < width && y < height) {
        int idx = y * width + x; 
        float4 a_ = cov_Ip[idx] / (var_I[idx] + eps);
        a[idx] = a_;
        b[idx] = mean_p[idx] - a_ * mean_I[idx];
    }
}

__device__ void compute_q(float4 *in, float4 *mean_a, float4 *mean_b, float4 *q,
        int width, int height)
{
    int x = blockIdx.x * TILE_W + threadIdx.x - RADIUS;
    int y = blockIdx.y * TILE_H + threadIdx.y - RADIUS;
    if (x >= 0 && y >= 0 && x < width && y < height) {
        int idx = y * width + x; 
        float4 im_ = in[idx];
        q[idx] = mean_a[idx] * im_ + mean_b[idx];
    }
}

__global__ void guidedFilterCudaKernel(float4* d_input,
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
        float4 *mean_a,
        float4 *mean_b,
        float4 *tmp,
        float4 *tmp2,
        int width, int height,
        float eps)
{
    mult(d_input, d_p, tmp, width, height);
    __syncthreads();
    mult(d_input, d_input, tmp2, width, height);

    __syncthreads();
    
    box_filter(d_input, mean_I, width, height);
    __syncthreads();
    box_filter(d_p, mean_p, width, height);
    __syncthreads();
    box_filter(tmp, mean_Ip, width, height);
    __syncthreads();
    box_filter(tmp2, mean_II, width, height);
    
    __syncthreads();

    compute_cov_var(mean_Ip, mean_II, mean_I, mean_p, var_I, cov_Ip, width, height);

    __syncthreads();
    
    compute_ab(var_I, cov_Ip, mean_I, mean_p, a, b, eps, width, height);
    
    __syncthreads();

    box_filter(a, mean_a, width, height);
    __syncthreads();
    box_filter(b, mean_b, width, height);

    __syncthreads();

    compute_q(d_p, mean_a, mean_b, d_q, width, height);
//    box_filter(d_input, d_q, width, height);
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

void guidedFilterCuda(float4 *h_input,
        float4 *h_p,
        float4 *h_output,
        int width, int height,
        float eps)
{
       
    const int n = width * height * sizeof(float4);

    float4 *d_input, *d_p, *d_output, *mean_I, *mean_p,* mean_Ip,
          *mean_II, *var_I, *cov_Ip, *a, *b, *mean_a, *mean_b, *tmp, *tmp2;
    checkCudaErrors(cudaMalloc<float4>(&d_input, n));
    checkCudaErrors(cudaMalloc<float4>(&d_p, n));
    checkCudaErrors(cudaMalloc<float4>(&d_output, n));
    checkCudaErrors(cudaMalloc<float4>(&mean_I, n));
    checkCudaErrors(cudaMalloc<float4>(&mean_p, n));
    checkCudaErrors(cudaMalloc<float4>(&mean_Ip, n));
    checkCudaErrors(cudaMalloc<float4>(&mean_II, n));
    checkCudaErrors(cudaMalloc<float4>(&var_I, n));
    checkCudaErrors(cudaMalloc<float4>(&cov_Ip, n));
    checkCudaErrors(cudaMalloc<float4>(&a, n));
    checkCudaErrors(cudaMalloc<float4>(&b, n));
    checkCudaErrors(cudaMalloc<float4>(&mean_a, n));
    checkCudaErrors(cudaMalloc<float4>(&mean_b, n));
    checkCudaErrors(cudaMalloc<float4>(&tmp, n));
    checkCudaErrors(cudaMalloc<float4>(&tmp2, n));

    checkCudaErrors(cudaMemcpy(d_input, h_input, n, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_p, h_p, n, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_output, h_output, n, cudaMemcpyHostToDevice));

    int GRID_W = width / TILE_W + 1;
    int GRID_H = height / TILE_H + 1;

    const dim3 block(BLOCK_W, BLOCK_H);
    const dim3 grid(GRID_W, GRID_H);
    //const dim3 grid(width/(block.x)+ block.x,height/(block.y)+block.y);
    //const dim3 grid((width + block.x-1)/block.x, (height + block.y - 1)/block.y);
    printf("grid_w: %d\n", grid.x);
    printf("grid_h: %d\n", grid.y);

    printf("block_w: %d\n", block.x);
    printf("block_h: %d\n", block.y);

    guidedFilterCudaKernel<<<grid, block>>>(d_input, d_p, d_output,
            mean_I, mean_p, mean_Ip, mean_II, var_I, cov_Ip, a, b,
            mean_a , mean_b, tmp, tmp2, width, height, eps);

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
    checkCudaErrors(cudaFree(tmp));
    checkCudaErrors(cudaFree(tmp2));
}

void processUsingCuda(std::string input_file, std::string output_file) {
    cv::Mat input = cv::imread(input_file);
    if(input.empty()) {
        std::cout<<"Image Not Found: "<< input_file << std::endl;
        return;
    }

    cv::Mat inputRGBA;
    cvtColor(input, inputRGBA, CV_BGR2RGBA, 4);
    inputRGBA.convertTo(inputRGBA, CV_32FC4);
    inputRGBA /= 255;

    cv::Mat p = inputRGBA.clone();

    cv::Mat output (input.size(), inputRGBA.type());

    float eps = 0.2 * 0.2;

    //std::cout << inputGRAY << std::endl;
    guidedFilterCuda((float4*)inputRGBA.ptr<float4>(),
            (float4*)p.ptr<float4>(),
            (float4*)output.ptr<float4>(),
            inputRGBA.cols, inputRGBA.rows,
            eps);

   // std::cout << output << std::endl;
    output *= 255;

    //output.convertTo(output, CV_32F);
    cvtColor(output, output, CV_RGBA2BGR, 3);

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
