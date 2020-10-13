#include "box_filter_grey.h"

#include<stdio.h>

__device__ void box_filter(float *in, float *out, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int idx = y * width + x;

    float sum = 0;
    int count = 0;

    for(int j = -4; j <= 4; j++) {
        for(int i = -4; i <= 4; i++) {
            if ((x + i) < width && (x + i) >= 0 && (y + j) < height && (y + j) >= 0) { 
                float tmp = in[((y + j) * width) + (x + i)];
                sum = sum + tmp;
                ++count;
            }
        }
    }
    out[idx] = sum / count ;
}
