#pragma once

#define TILE_W      16
#define TILE_H      16
#define R           8
#define D           (R*2+1)
#define S           (D*D)
#define BLOCK_W     (TILE_W+(2*R))
#define BLOCK_H     (TILE_H+(2*R))

__global__ void box_filter(float *in, float *out, int width, int height,
const int r);
