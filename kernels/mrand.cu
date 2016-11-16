#include "mrand.cuh"
__device__
uint32_t rand_xorshift(uint32_t& rng_state)
{
    // Xorshift algorithm from George Marsaglia's paper
    rng_state ^= (rng_state << 13);
    rng_state ^= (rng_state >> 17);
    rng_state ^= (rng_state << 5);
    return rng_state;
}

__device__
uint32_t wangHash(uint32_t seed)
{
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

// Generate a random float in [0, 1)...
__device__
double randDouble(uint32_t* seeds){
    
    uint32_t val = rand_xorshift(seeds[getGlobalIdx_2D_2D()]);
    return double(float(val) * 1.0/4294967296.0);
}

// Return a random Vec3d with each component in [0,1)
__device__
Vec3d randVec3d(uint32_t* seeds){
    return Vec3d(randDouble(seeds), randDouble(seeds), randDouble(seeds));
}

__device__
void initPNRG(uint32_t* seeds){
    int idxl = getGlobalIdx_2D_2D();
    seeds[idxl] = wangHash(idxl); //Initialize the seed
}
