#include <cuda.h>
#include <iostream>
#include <opencv2/opencv.hpp>

__constant__ float c_gaussian[64];

inline void computeGaussianKernelCuda(const float delta, const int radius)
{
	float h_gaussian[64];
	for (int i = 0; i < 2 * radius + 1; ++i)
	{
		const float x = i - radius;
		h_gaussian[i] = expf( -(x * x) / (2.0f * delta * delta) );
	}
	cudaMemcpyToSymbol(c_gaussian, h_gaussian, sizeof(float)*(2*radius+1));
}

__device__ inline float euclideanLenCuda(const float4 a, const float4 b, const float d)
{
	const float mod = (b.x - a.x) * (b.x - a.x) +
			(b.y - a.y) * (b.y - a.y) +
			(b.z - a.z) * (b.z - a.z) +
			(b.w - a.w) * (b.w - a.w);
	return expf(-mod / (2.0f * d * d));
}

__device__ inline float4 multiplyCuda(const float a, const float4 b)
{
	return {a * b.x, a * b.y, a * b.z, a * b.w};
}

__device__ inline float4 addCuda(const float4 a, const float4 b)
{
	return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}

__global__ void bilateralFilterCudaKernel( const float4 * const d_input,
		float4 * const d_output,
		const float euclidean_delta,
		const int width, const int height,
		const int filter_radius)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if((x<width) && (y<height))
	{
		float sum = 0.0f;
		float4 t = {0.f, 0.f, 0.f, 0.f};
		const float4 center = d_input[y * width + x];
		const int r = filter_radius;

		float domainDist=0.0f, colorDist=0.0f, factor=0.0f;

		for (int i = -r; i <= r; ++i)
		{
			int crtY = y + i;
			if (crtY < 0)
                crtY = 0;
			else if (crtY >= height)
                crtY = height - 1;

			for (int j = -r; j <= r; ++j)
			{
				int crtX = x + j;
				if (crtX < 0)
                    crtX = 0;
				else if (crtX >= width)
                    crtX = width - 1;

				const float4 curPix = d_input[crtY * width + crtX];
				domainDist = c_gaussian[r + i] * c_gaussian[r + j];
				colorDist = euclideanLenCuda(curPix, center, euclidean_delta);
				factor = domainDist * colorDist;
				sum += factor;
				t = addCuda(t, multiplyCuda(factor, curPix));
			}
		}

		d_output[y * width + x] = multiplyCuda(1.f / sum, t);
	}
}

void bilateralFilterCuda(const float4 * const h_input,
		float4 * const h_output,
		const float euclidean_delta,
		const int width, const int height,
		const int filter_radius)
{
	computeGaussianKernelCuda(euclidean_delta, filter_radius);

	const int inputBytes = width * height * sizeof(float4);
	const int outputBytes = inputBytes;
	float4 *d_input, *d_output;
	cudaMalloc<float4>(&d_input, inputBytes);
	cudaMalloc<float4>(&d_output, outputBytes);
	cudaMemcpy(d_input, h_input, inputBytes, cudaMemcpyHostToDevice);

	const dim3 block(16,16);
	const dim3 grid((width + block.x - 1)/block.x, (height + block.y - 1)/block.y);

	bilateralFilterCudaKernel<<<grid,block>>>(d_input, d_output, euclidean_delta, width, height, filter_radius);

	cudaDeviceSynchronize();

	cudaMemcpy(h_output, d_output, outputBytes, cudaMemcpyDeviceToHost);

	cudaFree(d_input);
	cudaFree(d_output);
}

void processUsingCuda(std::string input_file, std::string output_file) {
	cv::Mat input = cv::imread(input_file);
	if(input.empty())
	{
		std::cout<<"Image Not Found: "<< input_file << std::endl;
		return;
	}
 
    cv::Mat inputRGBA;
	cvtColor(input, inputRGBA, CV_BGR2RGBA, 4);
	inputRGBA.convertTo(inputRGBA, CV_32FC4);
	inputRGBA /= 255;
 
    cv::Mat output (input.size(), inputRGBA.type());
 
	const float euclidean_delta = 3.0f;
	const int filter_radius = 3;
 
	bilateralFilterCuda((float4*) inputRGBA.ptr<float4>(),
			(float4*) output.ptr<float4>(),
			euclidean_delta,
			inputRGBA.cols, inputRGBA.rows,
			filter_radius);
 
	output *= 255;
	cvtColor(output, output, CV_RGBA2BGR, 3);
 
	imwrite(output_file, output);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cout << "Choose image" << std::endl;
        return 1;
    }
    processUsingCuda("../data/cat.png", "out.png");
    return 0;
}
