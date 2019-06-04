#include "math_kernels.h"
#include "box_filter.h"

__device__ float4 operator*(float4 a, float4 b)
{
	return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

__device__ float4 operator/(float4 a, float4 b)
{
	return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}

__device__ float4 operator/(float4 a, int b)
{
    return make_float4(a.x / b, a.y / b, a.z / b,  a.w / b);
}

__device__ float4 operator-(float4 a, float4 b)
{
	return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

__device__ float4 operator+(float4 a, float4 b)
{
	return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__device__ float4 operator+(float4 a, float b)
{
	return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}

__device__ float4 fmaxf(float4 a, float b)
{
    return make_float4(fmaxf(a.x,b), fmaxf(a.y,b), fmaxf(a.z,b), fmaxf(a.w,b));
}

__device__ void mult(float *a, float *b, float *tmp, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 0 && y >= 0 && x < width && y < height) {
        int idx = y * width + x; 
        tmp[idx] = a[idx] * b[idx];
    }
}

__device__ void pown_ (float *in, float *out, int n,  int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 0 && y >= 0 && x < width && y < height) {
        int idx = y * width + x; 
        out[idx] = pow(in[idx], n);
    }
}
