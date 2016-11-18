#include "bvhl.cu"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#define OFFSET(i) i*numTriangles
#define MAX_DEPTH 20
#define MIN_ELEMS_PER_NODE 5

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

__global__
void initRoot(int depth, Node_L* nodes, double* sortedDims, int* sortedObjectIds, BSphere* treeBSpheres, BSphere* objBounds, BoundingBox* BBoxs, int numTriangles, Vec3d* lightOrigin){
    if(idx > 0)
        return;
    nodes[0].parent = 0;
    nodes[0].childA = 1;
    nodes[0].childB = 2;

    for(int i = 0; i < 3; i++){
        nodes[0].dmins[i] = 0;
        nodes[0].dmaxs[i] = numTriangles;
    }
}

__global__
void BuildHierarchy(int depth, Node_L* nodes, double* sortedDims, int* sortedObjectIds, BSphere* treeBSpheres, BSphere* objBounds, BoundingBox* BBoxs, int numTriangles, Vec3d* lightOrigin){

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= (1 << depth))
        return;

    int numElems = 0;
    for(int i = 0; i < 3; i++)
        numElems += nodes[MY_IDX(idx)].dmaxs[i] - nodes[MY_IDX(idx)].dmins[i];

    if(numElems < define MIN_ELEMS_PER_NODE || depth >= MAX_DEPTH){
        nodes[MY_IDX(idx)].childA = MY_IDX(idx);
        nodes[MY_IDX(idx)].childB = MY_IDX(idx);
        return;
    }

    int splitDim = 0;
    int splitLoc = 0;
    double minCost = 1.0e308;

    int k = 0;
    double* costL = new double[numElems];
    double* costR = new double[numElems];
    double projRadius = sortedDims[nodes[MY_IDX(idx)].dmaxs[0]];
    
    for(int i = i; i < 3; i++)
    {
        double solidAngle = ProjectionArea(*lightOrigin, projRadius, objBounds[sortedObjectIds[nodes[MY_IDX(idx)].dmins[i] + OFFSET(i)]]);
        costL[k] = solidAngle;
        costR[numElems - k - 1] = solidAngle;
        k++;
        for(int j = nodes[MY_IDX(idx)].dmins[i] + 1; j < nodes[MY_IDX(idx)].dmaxs[i]; j++)
        {
            //Project on the sphere centered on lightOrigin with projectionRadius defined at each level
            solidAngle = ProjectionArea(*lightOrigin, projRadius, objBounds[sortedObjectIds[j + OFFSET(i)]]);
            costL[k] = costL[k-1] + solidAngle;
            costR[numElems - k - 1] = costR[numElems - k] + solidAngle;
            k++;
        }// End inner for
    }

    // Find the Split Position
    k = 0;
    for(int i = 0; i < 3; i++){
        int numThingsInDim = nodes[MY_IDX(idx)].dmaxs[i] - nodes[MY_IDX(idx)].dmins[i];
        for(int j = nodes[MY_IDX(idx)].dmins[i], N = 1; j < nodes[MY_IDX(idx)].dmaxs[i]; j++, N++)
        {
            double currentCost = N*costL[k] + (numThingsInDim - N)*costR[k];
            if(currentCost < minCost){
                splitDim = i;
                splitLoc = j;
                minCost = currentCost;
            }

            k++;
        }
    }

    BoundingBox BLeft, BRight;
    //TODO FIXE THIS
    //Next line doesnt need to be recomputed
    //And pretty sure there are errors with the indexing
    //double splitVal = toSpherical(objBounds[sortedObjectId[splitLoc + OFFSET(splitDim)]], *lightOrigin)[splitDim];
    double splitVal = sortedDims[sortedObjectIds[splitLoc + OFFSET(splitDim)]];
    for(int i = 0; i < 3; i++){
        int numThingsInDim = nodes[MY_IDX(idx)].dmaxs[i] - nodes[MY_IDX(idx)].dmins[i];
        for(int j = nodes[MY_IDX(idx)].dmins[i], N = 1; j < nodes[MY_IDX(idx)].dmaxs[i]; j++, N++)
        {
            BoundingBox temp = BBoxs[sortedObjectIds[j + OFFSET(i)]];
            Vec3d sphereCoords = toSpherical(objBounds[sortedObjectIds[j + OFFSET(i)]].pos, *lightOrigin);
            //double sphereCoord = sortedDims[sortedObjectIds[j + OFFSET(i)]];
            if (sphereCoords[splitDim] <= splitVal)
            //if (sphereCoord <= splitVal)
                BLeft.merge(temp);
            else
                BRight.merge(temp);
        }

    }

    // Update Children
    int childA = FIND_CHILDA_OFFSET(idx);
    int childB = childA + 1;
    nodes[MY_IDX(idx)].childA = childA;
    nodes[MY_IDX(idx)].childB = childB;

    nodes[childA].parent = MY_IDX(idx);
    nodes[childB].parent = MY_IDX(idx);

    treeBSpheres[childA] = BSphere(BLeft.bmin, BLeft.bmax);
    treeBSpheres[childB] = BSphere(BRight.bmin, BRight.bmax);
    for(int i = 0; i < 3; i++){
        nodes[childA].dmins[i] = nodes[MY_IDX(idx)].dmins[i];
        nodes[childA].dmaxs[i] = nodes[MY_IDX(idx)].dmaxs[i];
        nodes[childB].dmins[i] = nodes[MY_IDX(idx)].dmins[i];
        nodes[childB].dmaxs[i] = nodes[MY_IDX(idx)].dmaxs[i];
    }
    
    nodes[childA].dmaxs[splitDim] = splitLoc;
    nodes[childB].dmins[splitDim] = splitLoc;

    delete[] costL;
    delete[] costR;
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

    // Initialize root
    
    //Build Hierarchy
    //+ 1 is done on purpose to Clean up leaf nodes
    for(int i = 0; i < MAX_DEPTH  + 1; i++)
    {

    }


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
