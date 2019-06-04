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
