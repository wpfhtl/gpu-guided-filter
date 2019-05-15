#include "box_filter.h"

#include <algorithm>

__global__ void box_filter(float *in, float *out, int width, int height)
{
    __shared__ unsigned char smem[BLOCK_W * BLOCK_H];
    int x = blockIdx.x * TILE_W + threadIdx.x - R;
    int y = blockIdx.y * TILE_H + threadIdx.y - R;

    x = std::max(0, x);
    x = std::min(x, width - 1);
    y = std::max(y, 0);
    y = std::min(y, height - 1);
    unsigned int index = y * width + x;
    unsigned int bindex = threadIdx.y * blockDim.y + threadIdx.x;
    smem[bindex] = in[index];

    __syncthreads();

    if ((threadIdx.x >= R) && (threadIdx.x < (BLOCK_W - R)) &&
        (threadIdx.y >= R) && (threadIdx.y < (BLOCK_H - R))) {
        float sum = 0;
        for(int dy = -R; dy <= r; dy++) {
            for(int dx = -R; dx <= r; dx++) {
                float i = smem[bindex + (dy * blockDim.x) + dx];
                sum += i;
            }
        }

        out[index] = sum / S;
    }
}
