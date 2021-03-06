#include "scene.h"
#include "debug.h"
#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>

struct minAccessor{
    
    __host__ __device__
    Vec3d operator () (const BoundingBox& a){
        return a.bmin;
    }
};

struct minFunctor{
    __host__ __device__
    Vec3d operator () (const Vec3d& a, const Vec3d& b){
        return minimum(a,b);
    }
};
struct maxAccessor{
    
    __host__ __device__
    Vec3d operator () (const BoundingBox& a){
        return a.bmax;
    }
};

struct maxFunctor{
    __host__ __device__
    Vec3d operator () (const Vec3d& a, const Vec3d& b){
        return maximum(a,b);
    }
};
// Declarations
__global__ 
void computeBoundingBoxes_kernel(int numTriangles, Vec3d* vertices, TriangleIndices* t_indices, BoundingBox* BBoxs);

__device__
BoundingBox computeTriangleBoundingBox(const Vec3d& a, const Vec3d& b, const Vec3d& c);

__global__
void AverageSuperSamplingKernel(Vec3d* smallImage, Vec3d* deviceImage, int imageWidth, int imageHeight, int superSampling);

//============================
//
void Scene_d::computeBoundingBoxes(){
    // Invoke kernel
    int N = numTriangles;
    int threadsPerBlock = 256;
    int blocksPerGrid =
        (N + threadsPerBlock - 1) / threadsPerBlock;
    computeBoundingBoxes_kernel<<<blocksPerGrid, threadsPerBlock>>>(numTriangles, vertices, t_indices, BBoxs);
    //cudaDeviceSynchronize();

}

void AverageSuperSampling(Vec3d* smallImage, 
                          Vec3d* deviceImage, 
                          int imageWidth, 
                          int imageHeight, 
                          int superSampling)
{
    int blockSize = 16;
    dim3 blockDim(blockSize, blockSize); //A thread block is 32x32 pixels
    dim3 gridDim(imageWidth/blockDim.x, imageHeight/blockDim.y);
    AverageSuperSamplingKernel<<<gridDim, blockDim>>>(smallImage, deviceImage, imageWidth, imageHeight, superSampling);
    cudaDeviceSynchronize();
}

void Scene_d::findMinMax(Vec3d& mMin, Vec3d& mMax){

    thrust::device_ptr<BoundingBox> dvp(BBoxs);
    mMin = thrust::transform_reduce(dvp, 
            dvp + numTriangles, 
            minAccessor(), 
            Vec3d(1e9, 1e9, 1e9), 
            minFunctor());
    mMax = thrust::transform_reduce(dvp, 
            dvp + numTriangles, 
            maxAccessor(),
            Vec3d(-1e9, -1e9, -1e9),
            maxFunctor());
}


__global__ 
void computeBoundingBoxes_kernel(int numTriangles, Vec3d* vertices, TriangleIndices* t_indices, BoundingBox* BBoxs){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numTriangles) return;

    TriangleIndices t_idx = t_indices[idx];
    BBoxs[idx] = computeTriangleBoundingBox(vertices[t_idx.a.vertex_index],vertices[t_idx.b.vertex_index],vertices[t_idx.c.vertex_index]);
    printf_DEBUG("idx(%d), a(%0.6f, %0.6f, %0.6f) b(%0.6f, %0.6f, %0.6f) c(%0.6f, %0.6f, %0.6f)\n" , idx, vertices[t_idx.a.vertex_index].x,
                                     vertices[t_idx.a.vertex_index].y,
                                     vertices[t_idx.a.vertex_index].z,
                                     vertices[t_idx.b.vertex_index].x,
                                     vertices[t_idx.b.vertex_index].y,
                                     vertices[t_idx.b.vertex_index].z,
                                     vertices[t_idx.c.vertex_index].x,
                                     vertices[t_idx.c.vertex_index].y,
                                     vertices[t_idx.c.vertex_index].z);

    printf_DEBUG("* T idx(%d) BB bmin(%0.6f,%0.6f,%0.6f) bmax(%0.6f,%0.6f,%0.6f) \n",idx, BBoxs[idx].bmin.x, BBoxs[idx].bmin.y, BBoxs[idx].bmin.z, BBoxs[idx].bmax.x, BBoxs[idx].bmax.y, BBoxs[idx].bmax.z);

    return;
}

__device__ 
BoundingBox computeTriangleBoundingBox(const Vec3d& a, const Vec3d& b, const Vec3d& c){
    BoundingBox bbox;
    BoundingBox localbounds;
    localbounds.setMax(maximum( a, b));
    localbounds.setMin(minimum( a, b));

    localbounds.setMax(maximum( c, localbounds.getMax()));
    localbounds.setMin(minimum( c, localbounds.getMin()));
    return localbounds;
}

//__device__
//bool Scene_d::intersect(const ray& r, isect& i){
//    return bvh.intersect(r, i, this);
//}

__global__
void AverageSuperSamplingKernel(Vec3d* smallImage, Vec3d* deviceImage, int imageWidth, int imageHeight, int superSampling)
{
    int pixelX = blockIdx.x*blockDim.x + threadIdx.x;
    int pixelY = blockIdx.y*blockDim.y + threadIdx.y;
    int pixelIdx = pixelY*imageWidth + pixelX;
    
    Vec3d mSum;
    if (superSampling == 1){
        mSum = deviceImage[pixelIdx];
        clamp(mSum);
        smallImage[pixelIdx] = mSum;
        return;
    }
    for(int i = 0; i < superSampling; i++){
         for(int j = 0; j < superSampling; j++){
            int idxX = pixelX*superSampling + j;
            int idxY = pixelY*superSampling + i;
            int idx = idxY*superSampling*imageWidth + idxX;
            Vec3d temp = deviceImage[idx];
            clamp(temp);
            mSum += temp; //Force it to be between 0 and 1
        }
    }

    //printf_DEBUG("idx = %d, mSum=%f,%f,%f\n", pixelIdx, mSum.x, mSum.y, mSum.z);
    mSum /= double(superSampling);
    smallImage[pixelIdx] = mSum;
}

