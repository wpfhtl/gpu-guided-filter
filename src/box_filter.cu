#include "box_filter.h"
#include<stdio.h>

__device__ void box_filter(float *in, float *out, int width, int height)
{
    int x = blockIdx.x * TILE_W + threadIdx.x - RADIUS;
    int y = blockIdx.y * TILE_H + threadIdx.y - RADIUS;
    const int idx = y * width + x;

    __shared__ float shMem[BLOCK_W][BLOCK_H];
    if(x<0 || y<0 || x>=width || y>=height) {
        shMem[threadIdx.x][threadIdx.y] = 0;
        return;
    }
    shMem[threadIdx.x][threadIdx.y] = in[idx];
   // printf("p_value: %f, im_value: %f, idx: %d\n", shMem[bindex], in[idx], idx);

    __syncthreads();

    if ((threadIdx.x >= RADIUS) && (threadIdx.x < (BLOCK_W - RADIUS)) &&
        (threadIdx.y >= RADIUS) && (threadIdx.y < (BLOCK_H - RADIUS))) {
        float sum = 0;
        for(int dy = -RADIUS; dy <= RADIUS; dy++) {
            for(int dx = -RADIUS; dx <= RADIUS; dx++) {
                float i = shMem[threadIdx.x + dx][threadIdx.y + dy];
                sum += i;
            }
        }
        out[idx] = sum / SIZE;
    }
}
