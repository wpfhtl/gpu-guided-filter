#pragma once

#define TILE_W      16
#define TILE_H      16
#define RADIUS      4
#define DIAM        (RADIUS*2+1)
#define SIZE        (DIAM*DIAM)
#define BLOCK_W     (TILE_W+(2*RADIUS))
#define BLOCK_H     (TILE_H+(2*RADIUS))

__device__ void box_filter(float *in, float *out, int width, int height);
