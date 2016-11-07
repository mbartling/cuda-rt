#pragma once
#include "scene.h"

//#define DIRECTIONAL_LIGHT 1
#define POSITIONAL_LIGHT 1

struct Light_h{

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
    float radius;
    //    f(d) = min( 1, 1/( a + b d + c d^2 ) )
    float constantTerm;        // a
    float linearTerm;        // b
    float quadraticTerm;    // c


    __host__ __device__
    Light_h(): color(1.0, 1.0, 1.0), position(1.0,1.0,1.0), orientation(.5, -1, -1), width(1) {} 
};
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
        
        float radius; // +-  from the center of the point (soft shadows)
        //    f(d) = min( 1, 1/( a + b d + c d^2 ) )
        float constantTerm;        // a
        float linearTerm;        // b
        float quadraticTerm;    // c

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
        __device__
            Light(Scene_d* scene, const Light_h& light): scene(scene) {
                color       = light.color       ;
                position    = light.position    ;
                //position    = Vec3f(1.0,1.0,1.0)    ;
                orientation = light.orientation ;
                width       = light.width       ;
                height      = light.height      ;
                constantTerm = 0.0;
                linearTerm = 0.01;
                quadraticTerm = 0.001;

            }

        //Directional Light
        __host__ __device__
            Light(Scene_d* scene, const Vec3f& col, const Vec3f& orientation) : color(col), scene(scene), orientation(orientation) {}

        //Point Light
        __host__ __device__
            Light(Scene_d* scene, const Vec3f& col, const Vec3f& position, const Vec3f& orientation) : color(col), scene(scene), position(position), orientation(orientation) {}

};
