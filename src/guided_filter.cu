#include "math_kernels.h"
#include "box_filter.h"
#include <cuda.h>

__global__ void gf_ab (float4 *mean_p, float4 *mean_p2, float4 *a, float4 *b,
                        float eps)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    
    float m_p = mean_p[x];
    float var_p = mean_p2[x] - pown (m_p, 2);
    float a_ = var_p / (var_p + eps);
    
    a[x] = a_;
    b[x] = (1.f - a_) * m_p;
}


__global__ void gf_var_Ip (float4 *corr_I, float4 *corr_Ip, float4 *mean_I,
                            float4 *mean_p, float4 *var_I, float4 *cov_Ip)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    
    float m_I = mean_I[x];

    var_I[x] = corr_I[x] - m_I * m_I;
    cov_Ip[x] = corr_Ip[x] - m_I * mean_p[x];
}

__global__ void gf_ab_Ip (float4 *var_I, float4 *cov_Ip, float4 *mean_I,
                            float4 *mean_p, float4 *a, float4 *b, float eps)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    
    float a_ = cov_Ip[x] / (var_I[x] + eps);
    
    a[x] = a_;
    b[x] = mean_p[x] - a_ * mean_I[x];
}

__global__ void gf_q (float4 *p, float4 *mean_a, float4 *mean_b, float4 *q,
                        int zero_out)
{

    int x = blockDim.x * blockIdx.x + threadIdx.x;

    float p_ = p[x];
    float q_ = mean_a[x] * p_ + mean_b[x];

    int4 p_select = isequal (p_, 0.f) * zero_out;
    float q_z = select(q_, 0.f, p_select);

    q[x] = scaling * q_z;
}

__global__ void guidedFilterCudaKernel( const float4 * const d_input,
		float4 * const d_p,
		float4 * const d_output,
		const int width, const int height,
	    float eps)
{

}

void guidedFilterCuda(const float4 * const h_input,
        const float4 * const h_p,
        float4 * const h_output,
		const int width, const int height,
		float4 eps)
{
	computeGaussianKernelCuda(euclidean_delta, filter_radius);

	const int inputBytes = width * height * sizeof(float4);
	const int outputBytes = inputBytes;
	float4 *d_input, *d_p, *d_output;
	cudaMalloc<float4>(&d_input, inputBytes);
	cudaMalloc<float4>(&d_p, inputBytes);
	cudaMalloc<float4>(&d_output, outputBytes);
	cudaMemcpy(d_input, h_input, inputBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_p, h_p, inputBytes, cudaMemcpyHostToDevice);

    int GRID_W = width / TILE_W + 1;
    int GRID_H = height / TILE_H + 1;
	const dim3 block(BLOCK_W, BLOCK_H);
	const dim3 grid(GRID_W, GRID_H);

	guidedFilterCudaKernel<<<grid,block>>>(d_input, d_p, d_output,
            euclidean_delta, width, height, eps);

	cudaDeviceSynchronize();

	cudaMemcpy(h_output, d_output, outputBytes, cudaMemcpyDeviceToHost);

	cudaFree(d_input);
	cudaFree(d_p);
	cudaFree(d_output);
}

void processUsingCuda(std::string input_file, std::string output_file) {
	cv::Mat input = cv::imread(input_file,IMREAD_UNCHANGED);
	if(input.empty())
	{
		std::cout<<"Image Not Found: "<< input_file << std::endl;
		return;
	}

	Mat inputRGBA;
	cvtColor(input, inputRGBA, CV_BGR2RGBA, 4);
	inputRGBA.convertTo(inputRGBA, CV_32FC4);
	inputRGBA /= 255;

    Mat p = inputRGBA.clone();

	Mat output (input.size(), inputRGBA.type());

	float eps = 0.2 * 0.2;

	guidedFilterCuda((float4*) inputRGBA.ptr<float4>(),
            (float4*) p.ptr<float4>(),
			(float4*) output.ptr<float4>(),
			inputRGBA.cols, inputRGBA.rows,
            eps);

	output *= 255;
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
