#include "material.h"
#include "ray.h"
#include "Light.h"

__device__
Vec3d Material::shade(Scene_d* scene, const ray& r, const isect& i) const
{
               Vec3d I = ke + (ka % scene->ambient());
               //Vec3d I = ke + ka;
    //Vec3d I = kd;

               Vec3d V = scene->getCamera()->getEye() - r.at(i.t) ;
               V = -V;
               normalize(V);

               Light* pLight  = scene->getLight();
               Vec3d lightDir = pLight->getDirection(r.at(i.t));
               ray toLightR(r.at(i.t), lightDir);
               //ray toLightR(r.at(i.t) + RAY_EPSILON*lightDir, lightDir);
               //ray toLightR(r.at(i.t) + i.N*RAY_EPSILON, lightDir);


               Vec3d atten    = pLight->distanceAttenuation(r.at(i.t)) * pLight->shadowAttenuation(toLightR, r.at(i.t));
               // Vec3d atten    = Vec3d() * pLight->shadowAttenuation(toLightR, r.at(i.t));
               double blah = i.N*lightDir;
               if(blah< 0) blah = 0;
               Vec3d diffuseTerm  = blah*kd;
               // Vec3d diffuseTerm  = maximum(Vec3d(0,0,0), i.N*lightDir*kd);


               Vec3d Rdir = -2.0*(lightDir*i.N)*i.N + lightDir;
               normalize(Rdir); 
               double tmp = Rdir*V;

               tmp =  powf(max(0.0, tmp), shininess);
               Vec3d specularTerm = tmp * ks;
               I += atten % ( diffuseTerm + specularTerm) % pLight->getColor();
    return I;
    // return kd;
}
