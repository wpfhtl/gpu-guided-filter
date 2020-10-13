#include "box_filter_grey.h"

#include<stdio.h>

__device__ void box_filter(float *in, float *out, int width, int height)
{
    __shared__ float smem[BLOCK_W*BLOCK_H];
    int x = blockIdx.x * TILE_W + threadIdx.x - RADIUS;
    int y = blockIdx.y * TILE_H + threadIdx.y - RADIUS;
    x = max(0, x);
    x = min(x, width-1);
    y = max(y, 0);
    y = min(y, height-1);
    const int idx = y * width + x;


    unsigned int bindex = threadIdx.y * blockDim.y + threadIdx.x;

    smem[bindex] = in[idx];
    __syncthreads();

    float sum = 0;
    int count = 0;
 
    if ((threadIdx.x >= RADIUS) && (threadIdx.x < (BLOCK_W - RADIUS)) &&
            (threadIdx.y >= RADIUS) && (threadIdx.y < (BLOCK_H - RADIUS))) {
       for(int dy = -RADIUS; dy <= RADIUS; dy++) {
            for(int dx = -RADIUS; dx <= RADIUS; dx++) {
                float i = smem[bindex + (dy * blockDim.x) + dx];
                sum = sum + i;
                ++count;
            }
        }
        out[idx] = sum / count;
    }
}
