#pragma once
#include "vec.h"
#include "indexing.h"

typedef unsigned int uint32_t;
__device__
uint32_t rand_xorshift(uint32_t& rng_state);

__device__
uint32_t wangHash(uint32_t seed);

// Generate a random float in [0, 1)...
__device__
double randDouble(uint32_t* seeds);

// Return a random Vec3d with each component in [0,1)
__device__
Vec3d randVec3d(uint32_t* seeds);

__device__
void initPNRG(uint32_t* seeds);
