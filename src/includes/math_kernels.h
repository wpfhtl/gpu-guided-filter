#pragma once

__device__ float4 operator*(float4 a, float4 b);
__device__ float4 operator/(float4 a, float4 b);
__device__ float4 operator/(float4 a, int b);
__device__ float4 operator-(float4 a, float4 b);
__device__ float4 operator+(float4 a, float4 b);
__device__ float4 operator+(float4 a, float b);
__device__ float4 fmaxf(float4 a, float b);
