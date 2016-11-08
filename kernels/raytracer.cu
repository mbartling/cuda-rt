#include "raytracer.h"
#include "debug.h"
#include "mrand.h"
#include "indexing.h"
#include <stdio.h>

#define PRINTVEC3(x) printf_DEBUG("(%f,%f,%f)",x.x,x.y,x.z)
struct RayStack{
    ray r;
    isect i;
    Vec3d colorC;
    int state;
};


__global__ 
void runRayTracerKernelRec(Scene_d* scene, int depth);

__device__ 
Vec3d traceRay(Scene_d* scene, ray& r, int depth);
__global__
void initLight(Scene_d* scene, Light_h hostLight, Light* light){
    printf_DEBUG("Adding Light to scene\n");
    *light = Light(scene, hostLight);
    scene->light = light;
}
void RayTracer::run(){
    int blockSize = 16;
    dim3 blockDim(blockSize, blockSize); //A thread block is 16x16 pixels
    dim3 gridDim(deviceScene.imageWidth/blockDim.x, deviceScene.imageHeight/blockDim.y);
    
    Scene_d* scene;
    Light* light;
    cudaMalloc(&scene, sizeof(Scene_d));
    cudaMalloc(&light, sizeof(Light));
    cudaMemcpy(scene, &deviceScene, sizeof(Scene_d), cudaMemcpyHostToDevice);

    initLight<<<1,1>>>(scene, hostLight, light);
    printf("\nLaunching Ray tracer kernel\n");
    cudaDeviceSynchronize();
    size_t stackSize;
    cudaDeviceSetLimit(cudaLimitStackSize, 1 << 16);
    cudaDeviceGetLimit(&stackSize, cudaLimitStackSize);
    printf_DEBUG("Stack size is %d\n", stackSize);
    runRayTracerKernelRec<<<gridDim, blockDim>>>(scene, depth);
    cudaDeviceSynchronize();
    printf("\nDone rendering Scene\n");
    
    cudaFree(scene); //Conveniently does not call destructor
    cudaFree(light);
}

__global__
void runRayTracerKernelRec(Scene_d* scene, int depth){

    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = (scene->imageHeight - py - 1)*scene->imageWidth + px;

    if (px >= scene->imageWidth)
        return;
    if (py >= scene->imageHeight)
        return;
    if (idx < 0 || idx >= scene->imageHeight * scene->imageWidth)
        return;
    //           double x = double(px)/double(scene->imageWidth);
    //           double y = double(py)/double(scene->imageHeight);

    // Initialize the PRNG
    initPNRG(scene->seeds);
    //Get view from the camera
    //perturb
    //x += randx; //in [0,1]
    //y += randy; //in [0,1]
    //           ray r;
    //           scene->getCamera()->rayThrough(x, y, r);
/*    double invWidth = 1.0 / double(scene->imageWidth), invHeight = 1.0 / double(scene->imageHeight);
    double fov = 35, aspectratio = double(scene->imageWidth) / double(scene->imageHeight);
    double angle = tan(M_PI * 0.5 * fov / 180.0f);
    double xx = (2 * ((px + 0.5) * invWidth) - 1)*angle*aspectratio;
    double yy = (1 - 2 * ((py + 0.5) * invHeight)) * angle;
    double focalDistance = 0.0433/(2.0*angle);
    //double focalDistance = 70.0/1000.0;
    double focalPoint = 7;
    //double lenseDistance = 1.0/(1.0/focalDistance - 1.0/focusPoint); //Doesnt matter
    double dofAngle = 2*M_PI*randDouble(scene->seeds);
    double dofRadius = scene->getCamera()->getAperature()*focalDistance * sqrt(randDouble(scene->seeds)) / 2.0;
    Vec3d origin(dofRadius*cos(dofAngle), dofRadius*sin(dofAngle), 0);
    //ray r(Vec3d(0.0,0.0,0.0), Vec3d(xx, yy, -1));
    ray r(origin, Vec3d(xx, yy, 1.0));
    r.d = origin - normalize(r.d)*focalPoint;
    normalize(r.d);
  */ 
    double invWidth = 1.0 / double(scene->imageWidth), invHeight = 1.0 / double(scene->imageHeight);
    double fov = 35, aspectratio = double(scene->imageWidth) / double(scene->imageHeight);
    double focalPoint = 7;
    double angle = tan(M_PI * 0.5 * fov / 180.0f);
    double xx = (2 * ((px + 0.5) * invWidth) - 1)*angle*aspectratio;
    double yy = (1 - 2 * ((py + 0.5) * invHeight)) * angle;
    double focalDistance = 0.0433/(2.0*angle);
    Vec3d colorC;
    int N = 5;
    //double focalDistance = 70.0/1000.0;
    //double lenseDistance = 1.0/(1.0/focalDistance - 1.0/focusPoint); //Doesnt matter
    for(int iter = 0; iter < N; iter++){
    double dofAngle = 2*M_PI*randDouble(scene->seeds);
    double dofRadius = scene->getCamera()->getAperature() * sqrt(randDouble(scene->seeds)) / 2.0;
    Vec3d origin(dofRadius*cos(dofAngle), dofRadius*sin(dofAngle), 0);
    //ray r(Vec3d(0.0,0.0,0.0), Vec3d(xx, yy, -1));
    ray r(origin, Vec3d(xx, yy, -1.0));

    r.d = normalize(r.d)*focalPoint - origin;
    normalize(r.d);
    
    printf_DEBUG("RAY %d, p=(%f,%f,%f), d=(%f,%f,%f)\n", idx, r.p.x, r.p.y, r.p.z, r.d.x, r.d.y, r.d.z);
    //printf_DEBUG("Attempting to trace ray\n");
    colorC += traceRay(scene, r, depth);
    }
    colorC /= double(N);
    scene->image[idx] = colorC;

}

__device__ 
Vec3d traceRay(Scene_d* scene, ray& r, int depth){
    isect* i = new isect();
    Vec3d colorC;

    // std::default_random_engine generator;
    // std::normal_distribution<double> distribution(0.0,0.01);
    if(scene->intersect(r, *i)) {
        // YOUR CODE HERE
        Vec3d q = r.at(i->t);

        //printf_DEBUG("q=(%f,%f,%f)\n", q.x, q.y, q.z);
        // An intersection occurred!  We've got work to do.  For now,
        // this code gets the material for the surface that was intersected,
        // and asks that material to provide a color for the ray.  

        // This is a great place to insert code for recursive ray tracing.
        // Instead of just returning the result of shade(), add some
        // more steps: add in the contributions from reflected and refracted
        // rays.
        const Material* m = &scene->materials[scene->material_ids[i->object_id]]; //i->material;	  
        colorC = m->shade(scene, r, *i);
        //        printf_DEBUG("colorC=(%f,%f,%f)\n", colorC.x, colorC.y, colorC.z);
        if(depth <= 0){
            delete i;
            return colorC;
        }
        /*
           if(m.Refl()){
        // std::cout<< "HERE"<< std::endl;

        Vec3d Rdir = -2.0*(r.getDirection()*i.N)*i.N + r.getDirection();
        normalize(Rdir);

        ray R(q, Rdir);
        colorC += m.kr % traceRay(scene, R, depth - 1);
        }
        // Now handle the Transmission (Refraction)
        if(m.Trans()){


        Vec3d n = i.N;
        Vec3d rd = r.getDirection();
        Vec3d rcos = n*(-rd*n);
        Vec3d rsin = rcos + rd;
        double etai = 1.0;
        double etat = m.ior;
        Vec3d tcos, tsin;
        double eta;
        if(rd*n < 0){
        eta = etai/etat;
        n = -n;
        } else{
        eta = etat/etai;
        }
        tsin = eta*rsin;
        double TIR = 1 - tsin*tsin;
        if(TIR >= 0){
        tcos = n*sqrt(TIR);
        Vec3d Tdir = tcos + tsin;
        normalize(Tdir);

        ray T(q, Tdir);
        colorC += m.kt % traceRay(scene, T, depth - 1);

        }

        }
        */

    } else {
        // No intersection.  This ray travels to infinity, so we color
        // it according to the background color, which in this (simple) case
        // is just black.
        //colorC = Vec3d(0.0, 0.0, 0.0);
        colorC = Vec3d(0.9, 0.9, 0.9);
    }
    delete i;
    return colorC;

}
