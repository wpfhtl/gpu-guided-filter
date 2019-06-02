#include "box_filter.h"
#include "math_kernels.h"

#include<stdio.h>

__device__ void box_filter(float4 *in, float4 *out, int width, int height)
{
    int x = blockIdx.x * TILE_W + threadIdx.x - RADIUS;
    int y = blockIdx.y * TILE_H + threadIdx.y - RADIUS;
    const int idx = y * width + x;

    __shared__ float4 shMem[BLOCK_W][BLOCK_H];
    if(x<0 || y<0 || x>=width || y>=height) {
        shMem[threadIdx.x][threadIdx.y] = make_float4(0, 0, 0, 0);
        return;
    }
    shMem[threadIdx.x][threadIdx.y] = in[idx];
   // printf("p_value: %f, im_value: %f, idx: %d\n", shMem[bindex], in[idx], idx);

    __syncthreads();

    if ((threadIdx.x >= RADIUS) && (threadIdx.x < (BLOCK_W - RADIUS)) &&
        (threadIdx.y >= RADIUS) && (threadIdx.y < (BLOCK_H - RADIUS))) {
        float4 sum = make_float4(0, 0, 0, 0);
        for(int dy = -RADIUS; dy <= RADIUS; dy++) {
            for(int dx = -RADIUS; dx <= RADIUS; dx++) {
                float4 i = shMem[threadIdx.x + dx][threadIdx.y + dy];
                sum = sum + i;
            }
        }
        out[idx] = sum / SIZE;
    }
}
