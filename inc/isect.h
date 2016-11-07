#pragma once
#include "vec.h"
#include "material.h"
class isect {
    public:
        double t;
        Vec3d N;
        Vec3d bary;
        int object_id;
//        Material material; // For smooth shading

//        __device__
//        Material& getMaterial() { return material; }

};
