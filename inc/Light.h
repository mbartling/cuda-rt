#pragma once
#include "scene.h"

#define DIRECTIONAL_LIGHT 1
// Currently a Directional Light
class Light {
    private:

        Scene_d* scene;
        Vec3f color;
//#if POINT_LIGHT
        Vec3f position;
//#endif
//#if DIRECTIONAL_LUGHT
        Vec3f orientation;
//#endif

        //Area Light Stuff
        float width;
        float height;

    public:
        __device__
        Vec3f shadowAttenuation(const ray& r, const Vec3f& pos);

        __device__
        float distanceAttenuation(const Vec3f& p);
        
        __device__
        Vec3f getDirection(const Vec3f& p);
        
        __device__
        Vec3f getColor();

        __host__ __device__
        Light(Scene_d* scene, const Vec3f& col) : color(col), scene(scene) {}
        
        //Directional Light
        __host__ __device__
        Light(Scene_d* scene, const Vec3f& col, const Vec3f& orientation) : color(col), scene(scene), orientation(orientation) {}
        
        //Point Light
        __host__ __device__
        Light(Scene_d* scene, const Vec3f& col, const Vec3f& position) : color(col), scene(scene), position(position) {}

};
