#include "raytracer.h"
#include <stdio.h>

#define PRINTVEC3(x) printf("(%f,%f,%f)",x.x,x.y,x.z)
struct RayStack{
    ray r;
    isect i;
    Vec3f colorC;
    int state;
};

__global__ 
void runRayTracerKernel(Scene_d scene, int depth);

__global__ 
void runRayTracerKernelRec(Scene_d* scene, int depth);

__device__ 
Vec3f traceRay(Scene_d* scene, ray& r, int depth);
__global__
void initLight(Scene_d* scene, Light_h hostLight, Light* light){
    printf("Adding Light to scene\n");
    *light = Light(scene, hostLight);
    scene->light = light;
}
void RayTracer::run(){
    int blockSize = 16;
    dim3 blockDim(blockSize, blockSize); //A thread block is 32x32 pixels
    dim3 gridDim(deviceScene.imageWidth/blockDim.x, deviceScene.imageHeight/blockDim.y);
    int stackDepth = ( 1 << depth) - 1;
    //runRayTracerKernel<<<gridDim, blockDim, stackDepth*sizeof(RayStack)>>>(deviceScene, depth);
    Scene_d* scene;
    Light* light;
    cudaMalloc(&scene, sizeof(Scene_d));
    cudaMalloc(&light, sizeof(Light));
    cudaMemcpy(scene, &deviceScene, sizeof(Scene_d), cudaMemcpyHostToDevice);

    initLight<<<1,1>>>(scene, hostLight, light);
    printf("\nABOUT TO RUN KERNEL\n");
    cudaDeviceSynchronize();
    size_t stackSize;
    cudaDeviceSetLimit(cudaLimitStackSize, 1 << 16);
    cudaDeviceGetLimit(&stackSize, cudaLimitStackSize);
    printf("Stack size is %d\n", stackSize);
    runRayTracerKernelRec<<<gridDim, blockDim>>>(scene, depth);
    cudaDeviceSynchronize();
    printf("Fuck THAT\n");
    cudaFree(scene);
    cudaFree(light);
}

__global__
void runRayTracerKernelRec(Scene_d* scene, int depth){

    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = py*scene->imageWidth + px;
    /*
       unsigned width = 640, height = 480; 
       Vec3f *image = new Vec3f[width * height], *pixel = image; 
       float invWidth = 1 / float(width), invHeight = 1 / float(height); 
       float fov = 30, aspectratio = width / float(height); 
       float angle = tan(M_PI * 0.5 * fov / 180.); 
    // Trace rays
    for (unsigned y = 0; y < height; ++y) { 
    for (unsigned x = 0; x < width; ++x, ++pixel) { 
    float xx = (2 * ((x + 0.5) * invWidth) - 1) * angle * aspectratio; 
    float yy = (1 - 2 * ((y + 0.5) * invHeight)) * angle; 
    Vec3f raydir(xx, yy, -1); 
    raydir.normalize(); 
     *pixel = trace(Vec3f(0), raydir, spheres, 0); 
    //                                                                                 } 
    //                                                                                     } i
    */
    //           float x = float(px)/float(scene->imageWidth);
    //           float y = float(py)/float(scene->imageHeight);

    //Get view from the camera
    //perturb
    //x += randx; //in [0,1]
    //y += randy; //in [0,1]
    //           ray r;
    //           scene->getCamera()->rayThrough(x, y, r);
    float invWidth = 1.0 / float(scene->imageWidth), invHeight = 1.0 / float(scene->imageHeight);
    float fov = 30, aspectratio = float(scene->imageWidth) / float(scene->imageHeight);
    float angle = tan(M_PI * 0.5 * fov / 180.0f);
    float xx = (2 * ((px + 0.5) * invWidth) - 1)*angle*aspectratio;
    float yy = (1 - 2 * ((py + 0.5) * invHeight)) * angle;
    ray r(Vec3f(0.0,0.0,0.0), Vec3f(xx, yy, -1));
    normalize(r.d);
    printf("RAY %d, p=(%f,%f,%f), d=(%f,%f,%f)\n", idx, r.p.x, r.p.y, r.p.z, r.d.x, r.d.y, r.d.z);
    Vec3f colorC;
    //printf("Attempting to trace ray\n");
    colorC = traceRay(scene, r, depth);

    scene->image[idx] = colorC;

}

__device__ 
Vec3f traceRay(Scene_d* scene, ray& r, int depth){
    isect* i = new isect();
    Vec3f colorC;

    // std::default_random_engine generator;
    // std::normal_distribution<float> distribution(0.0,0.01);
    if(scene->intersect(r, *i)) {
        // YOUR CODE HERE
        Vec3f q = r.at(i->t);

        //printf("q=(%f,%f,%f)\n", q.x, q.y, q.z);
        // An intersection occurred!  We've got work to do.  For now,
        // this code gets the material for the surface that was intersected,
        // and asks that material to provide a color for the ray.  

        // This is a great place to insert code for recursive ray tracing.
        // Instead of just returning the result of shade(), add some
        // more steps: add in the contributions from reflected and refracted
        // rays.
        const Material* m = &scene->materials[scene->material_ids[i->object_id]]; //i->material;	  
        colorC = m->shade(scene, r, *i);
        //        printf("colorC=(%f,%f,%f)\n", colorC.x, colorC.y, colorC.z);
        if(depth <= 0){
            delete i;
            return colorC;
        }
        /*
           if(m.Refl()){
        // std::cout<< "HERE"<< std::endl;

        Vec3f Rdir = -2.0*(r.getDirection()*i.N)*i.N + r.getDirection();
        normalize(Rdir);

        ray R(q, Rdir);
        colorC += m.kr % traceRay(scene, R, depth - 1);
        }
        // Now handle the Transmission (Refraction)
        if(m.Trans()){


        Vec3f n = i.N;
        Vec3f rd = r.getDirection();
        Vec3f rcos = n*(-rd*n);
        Vec3f rsin = rcos + rd;
        float etai = 1.0;
        float etat = m.ior;
        Vec3f tcos, tsin;
        float eta;
        if(rd*n < 0){
        eta = etai/etat;
        n = -n;
        } else{
        eta = etat/etai;
        }
        tsin = eta*rsin;
        float TIR = 1 - tsin*tsin;
        if(TIR >= 0){
        tcos = n*sqrt(TIR);
        Vec3f Tdir = tcos + tsin;
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
        colorC = Vec3f(0.0, 0.0, 0.0);
    }
    delete i;
    return colorC;

}
__global__ 
void runRayTracerKernel(Scene_d scene, int depth){
    extern __shared__ RayStack rayStack[];
    RayStack* stackPtr = rayStack;
    int curDepth = 0;

    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = py*scene.imageWidth + px;

    float x = float(px)/float(scene.imageWidth);
    float y = float(py)/float(scene.imageHeight);

    //Get view from the camera
    //perturb
    //x += randx; //in [0,1]
    //y += randy; //in [0,1]
    //scene.camera.rayThrough(x, y, stackPtr->r);

    /*
       while(true){
       ray& r = stackPtr->r;
       isect& i = stackPtr->i;
       Vec3f& colorC = stackPtr->colorC;
       int& state = stackPtr->state;

       if(state == 0) //Check for intersection
       {
       if(scene.intersect(r, i)){
       Vec3f q = r.at(i.t);
       colorC = i.material.shade(&scene, r, i);
       if(curDepth >= depth){state = 5;} //Exit
       else
       state = 1;
       }else{
       colorC = Vec3f(0.0,0.0,0.0);
       }
       }
       if(state == 1) //Check for reflection
       {
       if(!i.material.Refl())
       state = 3;
       else{
       Vec3f Rdir = -2.0*(r.getDirection()*i.N)*i.N + r.getDirection();
       normalize(Rdir);

    //Put DRT stuff HERE

    state = 2; //Select next state for my stack frame return
    Vec3f q = r.at(i.t);

    //Push
    stackPtr++;
    curDepth++;

    stackPtr->r = ray(q, Rdir);
    stackPtr->state = 0;
    continue; //Handle the stack push
    }
    }
    if(state == 2) //Post reflection
    {
    colorC += i.material.kr % (stackPtr+1)->colorC;
    state = 3;
    }
    if(state == 3) //Check for refraction
    {
    if(!i.material.Trans())
    state = 5; // Done
    else{
    Vec3f n = i.N;
    Vec3f rd = r.getDirection();
    Vec3f rcos = n*(-rd*n);
    Vec3f rsin = rcos + rd;
    float etai = 1.0;
    float etat = i.material.ior;
    Vec3f tcos, tsin;
    float eta;
    if(rd*n < 0){
    eta = etai/etat;
    n = -n;
    } else{
    eta = etat/etai;
    }
    tsin = eta*rsin;
    float TIR = 1 - tsin*tsin;
    if(TIR >= 0){
    tcos = n*sqrt(TIR);
    Vec3f Tdir = tcos + tsin;
    Vec3f q = r.at(i.t);
    normalize(Tdir);

    //Put DRT stuff HERE

    //Recusive part
    state = 4;

    //Push
    stackPtr++;
    curDepth++;

    stackPtr->r = ray(q, Tdir);
    stackPtr->state = 0;
    continue; //Handle the stack push
}
}
}
if(state == 4) //Post refraction
{

    colorC += i.material.kt % (stackPtr+1)->colorC;
    state = 5;
}
// There is no state 5 on purpose
if(curDepth == 0) //Hit nothing and am at root of stack
    break;
    else{
        stackPtr--; //Pop
        curDepth--;
    }

}

scene.image[idx] = rayStack[0].colorC;
*/
}
