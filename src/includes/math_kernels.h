#pragma once

__device__ float4 operator*(float4 a, float4 b);
__device__ float4 operator/(float4 a, float4 b);
__device__ float4 operator/(float4 a, int b);
__device__ float4 operator-(float4 a, float4 b);
__device__ float4 operator+(float4 a, float4 b);
__device__ float4 operator+(float4 a, float b);
__device__ float4 fmaxf(float4 a, float b);
__device__ void mult(float *a, float *b, float *tmp, int width, int height);
__device__ void pown_ (float *in, float *out, int n, int width, int height);
