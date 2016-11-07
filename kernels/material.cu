#include "material.h"
#include "ray.h"
#include "Light.h"

__device__
Vec3f Material::shade(Scene_d* scene, const ray& r, const isect& i) const
{
               Vec3f I = ke + (ka % scene->ambient());
               //Vec3f I = ke + ka;
    //Vec3f I = kd;

               Vec3f V = scene->getCamera()->getEye() - r.at(i.t) ;
               V = -V;
               normalize(V);

               Light* pLight  = scene->getLight();
               Vec3f lightDir = pLight->getDirection(r.at(i.t));
               ray toLightR(r.at(i.t), lightDir);


               Vec3f atten    = pLight->distanceAttenuation(r.at(i.t)) * pLight->shadowAttenuation(toLightR, r.at(i.t));
               // Vec3f atten    = Vec3f() * pLight->shadowAttenuation(toLightR, r.at(i.t));
               float blah = i.N*lightDir;
               if(blah< 0) blah = 0;
               Vec3f diffuseTerm  = blah*kd;
               // Vec3f diffuseTerm  = maximum(Vec3f(0,0,0), i.N*lightDir*kd);


               Vec3f Rdir = -2.0*(lightDir*i.N)*i.N + lightDir;
               normalize(Rdir); 
               float tmp = Rdir*V;

               tmp =  powf(max(0.0, tmp), shininess);
               Vec3f specularTerm = tmp * ks;
               I += atten % ( diffuseTerm + specularTerm) % pLight->getColor();
    return I;
    // return kd;
}
