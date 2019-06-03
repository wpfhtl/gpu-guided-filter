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

__device__ void mult(float4 *a, float4 *b, float4 *tmp, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 0 && y >= 0 && x < width && y < height) {
        int idx = y * width + x; 
        tmp[idx] = a[idx] * b[idx];
    }
}
/*
__device__ void pown_ (float4 *in, float4 *out, int n,  int width, int height)
{
    int x = blockIdx.x * TILE_W + threadIdx.x - RADIUS;
    int y = blockIdx.y * TILE_H + threadIdx.y - RADIUS;

    if (x < width && y < height) {
        int idx = y * width + x; 
        out[idx] = pow(in[idx], 2);
    }
}*/
