#include "box_filter.h"
#include "math_kernels.h"

#include<stdio.h>

__device__ void box_filter(float4 *in, float4 *out, int width, int height)
{
    __shared__ float4 smem[BLOCK_W*BLOCK_H];
    int x = blockIdx.x * TILE_W + threadIdx.x - RADIUS;
    int y = blockIdx.y * TILE_H + threadIdx.y - RADIUS;
    const int idx = y * width + x;


    unsigned int bindex = threadIdx.y * blockDim.y + threadIdx.x;

    smem[bindex] = in[idx];
    __syncthreads();

    float4 sum = make_float4(0, 0, 0, 0);
    int count = 0;
 
    if ((threadIdx.x >= RADIUS) && (threadIdx.x < (BLOCK_W - RADIUS)) &&
            (threadIdx.y >= RADIUS) && (threadIdx.y < (BLOCK_H - RADIUS))) {
       for(int dy = -RADIUS; dy <= RADIUS; dy++) {
            for(int dx = -RADIUS; dx <= RADIUS; dx++) {
                float4 i = smem[bindex + (dy * blockDim.x) + dx];
                sum = sum + i;
                ++count;
            }
        }
        out[idx] = sum / count;
    }
}
