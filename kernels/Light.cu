#include "Light.h"

__device__
Vec3d Light::shadowAttenuation(const ray& r, const Vec3d& pos){
#if POSITIONAL_LIGHT
    ray R = r;
    //R.p += RAY_EPSILON*R.d;
        //if(traceUI->isDistributionRayTracing()){
        //   Vec3d mDir = R.getDirection();
        //   Vec3d perterbT = CosWeightedRandomHemiDir2(mDir);
        //   R.d = mDir + perterbT;
        //   R.d.normalize();    
        // }

        isect* i = new isect();
    if(scene->intersect(R, *i)) { //We are potentially occluded
        if (i->t < norm(position - pos)){
            const Material* m = &scene->materials[scene->material_ids[i->object_id]]; //i->material;	  
            if(i->t < RAY_EPSILON){ delete i; return Vec3d(1,1,1);}

            if (m->Trans()){
                delete i;
                return m->kd % (Vec3d(1,1,1)- m->kt);
                //return m.kd(i) % m.kt(i);
            }
            else{
                delete i;
                return Vec3d(0,0,0);
            }
        }
        else{ delete i; return Vec3d(1,1,1);}
    }else{
        delete i;
        return Vec3d(1,1,1);
    }
#else //DIRECTIONAL_LIGHT
    ray R = r;
    isect* i = new isect();
    if (scene->intersect(R,*i)) { // We are occluded

        //const Material* m = i.getMaterial();
        const Material* m = &scene->materials[scene->material_ids[i->object_id]]; //i->material;	  
        if(i->t < RAY_EPSILON){ delete i; return Vec3d(1,1,1);}
        if (m->Trans()){delete i; return m->kd % (Vec3d(1,1,1) - m->kt);}
        else{ delete i; return Vec3d(0,0,0); }
    }else{
        delete i;
        return Vec3d(1,1,1);
    }
#endif
}

__device__
double Light::distanceAttenuation(const Vec3d& p){
#if POSITIONAL_LIGHT
    // You'll need to modify this method to attenuate the intensity 
    // of the light based on the distance between the source and the 
    // point P.  For now, we assume no attenuation and just return 1.0
    double d = norm(position - p);
    double falloff = constantTerm + (linearTerm + quadraticTerm*d)*d;
    falloff = (fabs(falloff) < RAY_EPSILON) ? 1.0 : 1.0 / falloff;
    return fmin( 1.0, falloff );
#else //DIRECTIONAL_LIGHT
    return 1.0; // Distance to light is infinite
#endif
}

__device__
Vec3d Light::getDirection(const Vec3d& p){
#if POSITIONAL_LIGHT
    Vec3d ret = position - p;
    normalize(ret);
    return ret;
#else //DIRECTIONAL_LIGHT
    return -orientation;
#endif
}

__device__
Vec3d Light::getColor(){
    return color;
}
