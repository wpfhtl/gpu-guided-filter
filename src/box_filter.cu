#include "box_filter.h"
#include "math_kernels.h"

#include<stdio.h>

__device__ void box_filter(float4 *in, float4 *out, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int idx = y * width + x;

    float4 sum = make_float4(0, 0, 0, 0);
    int count = 0;

    for(int j = -RADIUS; j <= RADIUS; j++) {
        for(int i = -RADIUS; i <= RADIUS; i++) {
            if ((x + i) < width && (x + i) >= 0 && (y + j) < height && (y + j) >= 0) { 
                float4 tmp = in[((y + j) * width) + (x + i)];
                sum = sum + tmp;
                ++count;
            }
        }
    }
    out[idx] = sum / (float)count ;
}
