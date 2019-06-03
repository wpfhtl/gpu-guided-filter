#pragma once

__device__ float4 operator*(float4 a, float4 b);
__device__ float4 operator/(float4 a, float4 b);
__device__ float4 operator/(float4 a, int b);
__device__ float4 operator-(float4 a, float4 b);
__device__ float4 operator+(float4 a, float4 b);
__device__ float4 operator+(float4 a, float b);
__device__ float4 fmaxf(float4 a, float b);
__device__ void mult(float4 *a, float4 *b, float4 *tmp, int width, int height);
//__device__ void pown_ (float4 *in, float4 *out, int n, int width, int height);
