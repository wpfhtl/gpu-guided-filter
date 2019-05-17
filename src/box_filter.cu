#include "box_filter.h"
#include<stdio.h>


__device__ void box_filter(float *in, float *out, int width, int height)
{
    __shared__ float shMem[BLOCK_W][BLOCK_H];
    int x = blockIdx.x * TILE_W + threadIdx.x - RADIUS;
    int y = blockIdx.y * TILE_H + threadIdx.y - RADIUS;
    x = max(0, x);
    x = min(x, width-1);
    y = max(y, 0);
    y = min(y, height-1);
    const int d = y * width + x;

    shMem[threadIdx.x][threadIdx.y] = in[d];

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
        printf("val: %f, sum: %f, size: %d\n", sum / SIZE, sum, SIZE);
        out[d] = sum / SIZE;
    }
}
