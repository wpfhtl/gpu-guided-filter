#include "math_kernels.h"

__global__ void mult (global float4 *a, global float4 *b, global float4 *out)
{
    int gX = blockDim.x * blockIdx.x + threadIdx.x;

	out[gX] = a[gX] * b[gX];
}

__global__ void pown_ (global float4 *in, global float4 *out, int n)
{
    int gX = blockDim.x * blockIdx.x + threadIdx.x;

	out[gX] = pown(in[gX], n);
}
