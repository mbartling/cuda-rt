#include "Light.h"

__device__
Vec3f Light::shadowAttenuation(const ray& r, const Vec3f& pos){
#if DIRECTIONAL_LIGHT
    ray R = r;
    isect i;
    if (scene->intersect(R,i)) { // We are occluded

        const Material& m = i.getMaterial();
        if(i.t < RAY_EPSILON) return Vec3f(1,1,1);
        if (m.Trans()){return m.kd % (Vec3f(1,1,1) - m.kt);}
        else return Vec3f(0,0,0);
    }else{
        return Vec3f(1,1,1);
    }
#endif
}

__device__
float Light::distanceAttenuation(const Vec3f& p){
#if DIRECTIONAL_LIGHT
    return 1.0; // Distance to light is infinite
#endif
}

__device__
Vec3f Light::getDirection(const Vec3f& p){
#if DIRECTIONAL_LIGHT
    return -orientation;
#endif
}

__device__
Vec3f Light::getColor(){
    return color;
}
