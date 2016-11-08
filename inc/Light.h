#pragma once
#include "scene.h"

//#define DIRECTIONAL_LIGHT 1
#define POSITIONAL_LIGHT 1

struct Light_h{

    Vec3d color;
    //#if POINT_LIGHT
    Vec3d position;
    //#endif
    //#if DIRECTIONAL_LUGHT
    Vec3d orientation;
    //#endif

    //Area Light Stuff
    double width;
    double height;
    double radius;
    //    f(d) = min( 1, 1/( a + b d + c d^2 ) )
    double constantTerm;        // a
    double linearTerm;        // b
    double quadraticTerm;    // c


    __host__ __device__
    Light_h(): color(1.0, 1.0, 1.0), position(1.0,1.0,1.0), orientation(.5, -1, -1), width(1) {} 
};
// Currently a Directional Light
class Light {
    private:

        Scene_d* scene;
        Vec3d color;
        //#if POINT_LIGHT
        Vec3d position;
        //#endif
        //#if DIRECTIONAL_LUGHT
        Vec3d orientation;
        //#endif

        //Area Light Stuff
        double width;
        double height;
        
        double radius; // +-  from the center of the point (soft shadows)
        //    f(d) = min( 1, 1/( a + b d + c d^2 ) )
        double constantTerm;        // a
        double linearTerm;        // b
        double quadraticTerm;    // c

    public:
        __device__
            Vec3d shadowAttenuation(const ray& r, const Vec3d& pos);

        __device__
            double distanceAttenuation(const Vec3d& p);

        __device__
            Vec3d getDirection(const Vec3d& p);

        __device__
            Vec3d getColor();

        __host__ __device__
            Light(Scene_d* scene, const Vec3d& col) : color(col), scene(scene) {}
        __device__
            Light(Scene_d* scene, const Light_h& light): scene(scene) {
                color       = light.color       ;
                position    = light.position    ;
                //position    = Vec3d(1.0,1.0,1.0)    ;
                orientation = light.orientation ;
                width       = light.width       ;
                height      = light.height      ;
                constantTerm = 0.0;
                linearTerm = 0.0;
                quadraticTerm = 0.01;

            }

        //Directional Light
        __host__ __device__
            Light(Scene_d* scene, const Vec3d& col, const Vec3d& orientation) : color(col), scene(scene), orientation(orientation) {}

        //Point Light
        __host__ __device__
            Light(Scene_d* scene, const Vec3d& col, const Vec3d& position, const Vec3d& orientation) : color(col), scene(scene), position(position), orientation(orientation) {}

};
