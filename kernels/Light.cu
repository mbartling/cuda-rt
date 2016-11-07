#include "Light.h"

__device__
Vec3f Light::shadowAttenuation(const ray& r, const Vec3f& pos){
#if DIRECTIONAL_LIGHT
    ray R = r;
    isect* i = new isect();
    if (scene->intersect(R,*i)) { // We are occluded

        //const Material* m = i.getMaterial();
        const Material* m = &scene->materials[scene->material_ids[i->object_id]]; //i->material;	  
        if(i->t < RAY_EPSILON){ delete i; return Vec3f(1,1,1);}
        if (m->Trans()){delete i; return m->kd % (Vec3f(1,1,1) - m->kt);}
        else{ delete i; return Vec3f(0,0,0); }
    }else{
        delete i;
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
