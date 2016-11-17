#include "bvhl.cu"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#define OFFSET(i) i*numTriangles
__host__ __device__
Vec3d toSpherical(const Vec3d& pos, const Vec3d& lightOrigin){
    double r = norm(pos - lightOrigin);
    double theta = atan2(pos.y, pos.x);
    double phi = acos(pos.z / r);
    
    return Vec3d(r,theta,phi);
}
__global__
void setupBounds(Vec3d* lightOrigin, 
        Vec3d mMin, 
        Vec3d mMax, 
        BoundingBox* BBoxs, 
        Vec3d* sortedDims, 
        Int3Ids* sortedObjectIds, 
        BSphere* objBounds, 
        int numTriangles)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >=  numTriangles)
        return;

    sortedObjectIds[idx + OFFSET(0)] = idx;
    sortedObjectIds[idx + OFFSET(1)] = idx;
    sortedObjectIds[idx + OFFSET(2)] = idx;

    objBounds[idx] = BSphere(BBoxs[idx].bmin, BBoxs[idx].bmax);

    Vec3d spCoords = toSpherical(objBounds[idx].pos, *lightOrigin);

    sortedDims[idx + OFFSET(0)] = spCoords.x;
    sortedDims[idx + OFFSET(1)] = spCoords.y;
    sortedDims[idx + OFFSET(2)] = spCoords.z;
}

void BVH_L::sortDimensions(){
    thrust::device_ptr<double> dev_Dims(sortedDims);
    thrust::device_ptr<int> dev_object_ids(sortedObjectIds);

    // Let thrust do all the work for us
    for( int i = 0; i < 3; i++)
        thrust::sort_by_key(dev_Dims + i*numTriangles, dev_Dims + (i+1)*numTriangles, dev_object_ids + i*numTriangles);


}

void BVH_L::buildTree(Vec3d mMin, Vec3d mMax){
    int threadsPerBlock = 256;
    int blocksPerGrid =
        (numTriangles + threadsPerBlock - 1) / threadsPerBlock;

    // Initialize
    setupBounds<<<blocksPerGrid, threadsPerBlock>>>(
        lightOrigin, 
        mMin, 
        mMax, 
        BBoxs, 
        sortedDims, 
        sortedObjectIds, 
        objBounds, 
        numTriangles);
    
    // Now Sort the dimensions
    sortDimensions();


}


__device__
bool BVH_L::intersectTriangle(const ray& r, isect&  i, int object_id) const
{

    const TriangleIndices* ids = &t_indices[object_id];

    const Vec3d* a = &vertices[ids->a.vertex_index];
    const Vec3d* b = &vertices[ids->b.vertex_index];
    const Vec3d* c = &vertices[ids->c.vertex_index];

    /*
       -DxAO = AOxD
       AOx-D = -(-DxAO)
       |-D AB AC| = -D*(ABxAC) = -D*normal = 1. 1x
       |AO AB AC| = AO*(ABxAC) = AO*normal = 1. 
       |-D AO AC| = -D*(AOxAC) = 1. 1x || AC*(-DxAO) = AC*M = 1. 1x
       |-D AB AO| = -D*(ABxAO) = 1. 1x || (AOx-D)*AB = (DxAO)*AB = -M*AB = 1.
       */
    double mDet;
    double mDetInv;
    double alpha;
    double beta;
    double t;
    //Moller-Trombore approach is a change of coordinates into a local uv space
    // local to the triangle
    Vec3d AB = *b - *a;
    Vec3d AC = *c - *a;

    Vec3d normal = AB ^ AC;
    // if (normal * -r.getDirection() < 0) return false;
    Vec3d P = r.getDirection() ^ AC;
    mDet = AB * P;
#if WITH_CULLING
    if(mDet < RAY_EPSILON ) return false;   //With culling
#else 
    if(fabsf(mDet) < RAY_EPSILON ) return false; //Normal
#endif
    //if(mDet > -RAY_EPSILON && mDet < RAY_EPSILON ) return false; //Normal

    mDetInv = 1/mDet;
    Vec3d T = r.getPosition() - *a;
    alpha = T * P * mDetInv;    
    if(alpha < 0 || alpha > 1) return false;

    Vec3d Q = T ^ AB;
    beta = r.getDirection() * Q * mDetInv;
    if(beta < 0 || alpha + beta > 1) return false;
    t = AC * Q * mDetInv;

    //if( t < RAY_EPSILON ) return false;
    if(fabsf(t) < RAY_EPSILON) return false; // Jaysus this sucked
    //i.bary = Vec3d(1 - (alpha + beta), alpha, beta);
    //printf_DEBUG2("t=%f\n", t);
    //if( t < 0.0 ) return false;
    i.t = t;


    // std::cout << traceUI->smShadSw() << std::endl; 
    // if(traceUI->smShadSw() && !parent->doubleCheck()){
    //Smooth Shading
    i.N = (1.0 - (alpha + beta))*normals[ids->a.normal_index] + \
          alpha*normals[ids->b.normal_index] + \
          beta*normals[ids->c.normal_index];
    //i.N = normal;

    //i.N = normal;

    normalize(i.N);

    i.object_id = object_id;

    return true;


}
