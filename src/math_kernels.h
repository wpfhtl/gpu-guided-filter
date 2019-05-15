#pragma once

__global__ void mult (global float4 *a, global float4 *b, global float4 *out);
__global__ void pown_ (global float4 *in, global float4 *out, int n);
