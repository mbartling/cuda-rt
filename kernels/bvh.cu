#include <stdio.h>
#include "bvh.cuh"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include "scene.h"
#include "debug.h"


// The following two functions are from
// // http://devblogs.nvidia.com/parallelforall/thinking-parallel-part-iii-tree-construction-gpu/
// // Expands a 10-bit integer into 30 bits
// // by inserting 2 zeros after each bit.
    __device__
unsigned int ExpandBits(unsigned int v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}
    __device__
unsigned int CalculateMortonCode(Vec3d p)
{
    float x = (float)min(max(p.x * 1024.0f, 0.0f), 1023.0f);
    float y = (float)min(max(p.y * 1024.0f, 0.0f), 1023.0f);
    float z = (float)min(max(p.z * 1024.0f, 0.0f), 1023.0f);
    unsigned int xx = ExpandBits((unsigned int)x);
    unsigned int yy = ExpandBits((unsigned int)y);
    unsigned int zz = ExpandBits((unsigned int)z);
    return xx * 4 + yy * 2 + zz;
}


// This kernel just computes the object id and morton code for the centroid of each bounding box
__global__ 
void computeMortonCodesKernel(int* mortonCodes, unsigned int* object_ids, 
        BoundingBox* BBoxs, TriangleIndices* t_indices, Vec3d* vertices, int numTriangles, Vec3d mMin , Vec3d mMax){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numTriangles)
        return;

    object_ids[idx] = idx;
    /*
       const TriangleIndices* ids = &t_indices[idx];

       Vec3d a = vertices[ids->a.vertex_index];
       Vec3d b = vertices[ids->b.vertex_index];
       Vec3d c = vertices[ids->c.vertex_index];
       Vec3d centroid = (a + b + c)/3.0;//= computeCentroid(BBoxs[idx]);
       */
    BoundingBox bound = BBoxs[idx];
    Vec3d centroid = 0.5*(bound.bmin + bound.bmax);
    centroid.x = (centroid.x - mMin.x)/(mMax.x - mMin.x);
    centroid.y = (centroid.y - mMin.y)/(mMax.y - mMin.y);
    centroid.z = (centroid.z - mMin.z)/(mMax.z - mMin.z);
    //map this centroid to unit cube
    mortonCodes[idx] = CalculateMortonCode(centroid);//morton3D(centroid.x, centroid.y, centroid.z);
    printf_DEBUG("in computeMortonCodesKernel: idx->%d , mortonCode->%d, centroid(%0.6f,%0.6f,%0.6f)\n", idx, mortonCodes[idx], centroid.x, centroid.y, centroid.z);

};

__device__
BoundingBox bboxunion(BoundingBox b1, BoundingBox b2){
    BoundingBox res;
    res.bmin = minimum(b1.bmin, b2.bmin);
    res.bmax = maximum(b1.bmax, b2.bmax);
    return res;
}

__device__
int delta(int* mortoncodes, int numprims, int i1, int i2){
    // Select left end
    int left = min(i1, i2);
    // Select right end
    int right = max(i1, i2);
    // This is to ensure the node breaks if the index is out of bounds
    if (left < 0 || right >= numprims) 
    {
        return -1;
    }
    // Fetch Morton codes for both ends
    int leftcode = mortoncodes[left];
    int rightcode = mortoncodes[right];

    // Special handling of duplicated codes: use their indices as a fallback
    return leftcode != rightcode ? __clz(leftcode ^ rightcode) : (32 + __clz(left ^ right));
}

#define DELTA(i,j) delta(mortoncodes,numprims,i,j)

__device__
int sign(float x){
    if(x > 0) return 1;
    else if(x == -0.0) return -0;
    else if(x == 0.0) return 0;
    else if(x < 0) return -1;
    else return 0; //NAN case
}
__device__ 
int2 FindSpan(int* mortoncodes, int numprims, int idx){
    // Find the direction of the range
    int d = sign((float)(DELTA(idx, idx+1) - DELTA(idx, idx-1)));

    // Find minimum number of bits for the break on the other side
    int deltamin = DELTA(idx, idx-d);

    // Search conservative far end
    int lmax = 2;
    while (DELTA(idx,idx + lmax * d) > deltamin)
        lmax *= 2;

    // Search back to find exact bound
    // with binary search
    int l = 0;
    int t = lmax;
    do
    {
        t /= 2;
        if(DELTA(idx, idx + (l + t)*d) > deltamin)
        {
            l = l + t;
        }
    }
    while (t > 1);

    // Pack span 
    int2 span;
    span.x = min(idx, idx + l*d);
    span.y = max(idx, idx + l*d);
    return span;

}
    __device__
int FindSplit(int* mortoncodes, int numprims, int2 span)
{
    // Fetch codes for both ends
    int left = span.x;
    int right = span.y;

    // Calculate the number of identical bits from higher end
    int numidentical = DELTA(left, right);

    do
    {
        // Proposed split
        int newsplit = (right + left) / 2;

        // If it has more equal leading bits than left and right accept it
        if (DELTA(left, newsplit) > numidentical)
        {
            left = newsplit;
        }
        else
        {
            right = newsplit;
        }
    }
    while (right > left + 1);

    return left;
}

__global__
void BuildHierarchy(int* mortoncodes, BoundingBox* bounds, unsigned int* indices, int numprims, HNode* nodes, BoundingBox* boundssorted, int* flags){

    int globalid = blockIdx.x * blockDim.x + threadIdx.x;
    // Set child
    if (globalid < numprims)
    {
        nodes[LEAFIDX(globalid)].left = nodes[LEAFIDX(globalid)].right = indices[globalid];
        boundssorted[LEAFIDX(globalid)] = bounds[indices[globalid]];
    }

    // Set internal nodes
    if (globalid < numprims - 1)
    {
        flags[NODEIDX(globalid)] = 0;
        // Find span occupied by the current node
        int2 range = FindSpan(mortoncodes, numprims, globalid);

        // Find split position inside the range
        int  split = FindSplit(mortoncodes, numprims, range);

        // Create child nodes if needed
        int c1idx = (split == range.x) ? LEAFIDX(split) : NODEIDX(split);
        int c2idx = (split + 1 == range.y) ? LEAFIDX(split + 1) : NODEIDX(split + 1);

        nodes[NODEIDX(globalid)].left = c1idx;
        nodes[NODEIDX(globalid)].right = c2idx;
        //nodes[NODEIDX(globalid)].next = (range.y + 1 < numprims) ? range.y + 1 : -1;
        nodes[c1idx].parent = NODEIDX(globalid);
        nodes[c2idx].parent = NODEIDX(globalid);
        //if(globalid == 0){
        //printf_DEBUG2("IDX(%d) L(%d), R(%d), range=[%d,%d], sp(%d)\n",globalid, TOLEAFIDX(c1idx),TOLEAFIDX(c2idx), range.x, range.y, split);
        if(LEAFNODE(nodes[c1idx]) && LEAFNODE(nodes[c2idx]))
            printf_DEBUG2("IDX(%d) L_LN(%d), R_LN(%d), range=[%d,%d], sp(%d)\n",globalid, TOLEAFIDX(c1idx),TOLEAFIDX(c2idx), range.x, range.y, split);
        else if(LEAFNODE(nodes[c1idx]))
         printf_DEBUG2("IDX(%d) L_LN(%d), R_IN(%d), range=[%d,%d], sp(%d)\n",globalid, TOLEAFIDX(c1idx), c2idx, range.x, range.y, split);
        else if(LEAFNODE(nodes[c2idx]))
            printf_DEBUG2("IDX(%d) L_IN(%d), R_LN(%d), range=[%d,%d], sp(%d)\n",globalid, c1idx, TOLEAFIDX(c2idx), range.x, range.y, split);
        else
            printf_DEBUG2("IDX(%d) L_IN(%d), R_IN(%d), range=[%d,%d], sp(%d)\n",globalid, c1idx, c2idx, range.x, range.y, split);
         //printf_DEBUG2("IDX(%d) L(%d), R(%d), range=[%d,%d], sp(%d)\n",globalid, c1idx, c2idx, range.x, range.y, split);
       // }
    }
}

__global__
void RefitBounds(BoundingBox* bounds, int numprims, HNode* nodes, int* flags){
    int globalid = blockIdx.x * blockDim.x + threadIdx.x;

    // Start from leaf nodes
    if (globalid < numprims)
    {
        // Get my leaf index
        int idx = LEAFIDX(globalid);
        int* flagidx;

        do
        {
            // Move to parent node
            idx = nodes[idx].parent;
            flagidx = flags + idx;

            //__threadfence();
            // Check node's flag
            if (atomicCAS(flagidx, 0, 1))
            {
                // If the flag was 1 the second child is ready and 
                // this thread calculates bbox for the node

                // Fetch kids
                int lc = nodes[idx].left;
                int rc = nodes[idx].right;

                // Calculate bounds
                BoundingBox b = bboxunion(bounds[lc], bounds[rc]);

                // Write bounds
                bounds[idx] = b;
                printf_DEBUG2("idx(%d) L=%d, R=%d bmin(%f,%f,%f),bmax(%f,%f,%f)\n",idx,lc,rc, b.bmin.x, b.bmin.y, b.bmin.z, b.bmax.x, b.bmax.y, b.bmax.z);
            }
            else
            {
                int lc = nodes[idx].left;
                int rc = nodes[idx].right;
                printf_DEBUG2("Got here First idx(%d) L=%d, R=%d\n",idx,lc,rc);
                // If the flag was 0 set it to 1 and bail out.
                // The thread handling the second child will
                // handle this node.
                break;
            }
        }
        while (idx != 0);
    }

}

void BVH_d::computeMortonCodes(Vec3d& mMin, Vec3d& mMax){
    int threadsPerBlock = 256;
    int blocksPerGrid =
        (numTriangles + threadsPerBlock - 1) / threadsPerBlock;
    computeMortonCodesKernel<<<blocksPerGrid, threadsPerBlock>>>(mortonCodes, object_ids, BBoxs, t_indices, vertices, numTriangles, mMin , mMax);

}
void BVH_d::sortMortonCodes(){
    thrust::device_ptr<int> dev_mortonCodes(mortonCodes);
    thrust::device_ptr<unsigned int> dev_object_ids(object_ids);

    // Let thrust do all the work for us
    thrust::sort_by_key(dev_mortonCodes, dev_mortonCodes + numTriangles, dev_object_ids);
}

void BVH_d::setupLeafNodes(){
    //Does nothing
}
void BVH_d::buildTree(){
    int threadsPerBlock = 256;
    int blocksPerGrid = (numTriangles + threadsPerBlock - 1) / threadsPerBlock;
    //int blocksPerGrid = (numTriangles - 1 + threadsPerBlock - 1) / threadsPerBlock;

    cudaDeviceSynchronize();
    BuildHierarchy<<<blocksPerGrid, threadsPerBlock>>>(mortonCodes, BBoxs, object_ids, numTriangles, nodes, sortedBBoxs, flags);
    cudaDeviceSynchronize();
    std::cout << "Post Generate Hierarchy " << cudaGetErrorString(cudaGetLastError()) << std::endl;

    RefitBounds<<<blocksPerGrid, threadsPerBlock>>>(sortedBBoxs, numTriangles, nodes, flags);
    cudaDeviceSynchronize();
    std::cout << "Post compute Tree BBoxes " << cudaGetErrorString(cudaGetLastError()) << std::endl;

}


//===========Begin KERNELS=============================
//===========Begin KERNELS=============================
__device__
bool BVH_d::intersectTriangle(const ray& r, isect&  i, int object_id) const
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

__device__
void printBBox(const BoundingBox& b){
    printf_DEBUG1("[bmin(%f,%f,%f) bmax(%f,%f,%f)]", b.bmin.x, b.bmin.y, b.bmin.z,b.bmax.x, b.bmax.y, b.bmax.z);
}

#define STARTIDX(x)     (((int)((x).left)))
#define STACK_SIZE 64
#define SHORT_STACK_SIZE 16

__device__
bool BVH_d::intersect(const ray& r, isect& i) const{
    int stack[STACK_SIZE];
    int topIndex = STACK_SIZE;
    stack[--topIndex] = 0; // Get root
    bool haveOne = false;
    isect cur;// = new isect();
    while (topIndex != STACK_SIZE){
        int nodeIdx = stack[topIndex++];
        if(sortedBBoxs[nodeIdx].intersect(r)) {
            if (LEAFNODE(nodes[nodeIdx])) {
                if(intersectTriangle(r, cur, nodes[nodeIdx].left)){ //Set this in the build
                    if((!haveOne || (cur.t < i.t))){
                        //if((!haveOne || (cur->t < i.t)) && cur->t > RAY_EPSILON){
                        i = cur;
                        haveOne = true;
                    }
                }

            } else{
                stack[--topIndex] = nodes[nodeIdx].right;
                stack[--topIndex] = nodes[nodeIdx].left;
                if(topIndex < 0){
                    printf_DEBUG2("Intersect stack not big enough!\n");
                    return false;
                }
            }
        }

    }
        //delete cur;
        return haveOne;
}

__device__
bool BVH_d::intersectAny(const ray& r, isect& i) const{
    int stack[STACK_SIZE];
    int topIndex = STACK_SIZE;
    stack[--topIndex] = 0; // Get root
    bool haveOne = false;
    isect cur;// = new isect();
    while (topIndex != STACK_SIZE){
        int nodeIdx = stack[topIndex++];
        if(sortedBBoxs[nodeIdx].intersect(r)) {
            if (LEAFNODE(nodes[nodeIdx])) {
                if(intersectTriangle(r, cur, nodes[nodeIdx].left)){ //Set this in the build
                    if((!haveOne || (cur.t < i.t))){
                        //if((!haveOne || (cur->t < i.t)) && cur->t > RAY_EPSILON){
                        i = cur;
                        haveOne = true;
                        return true;
                    }
                }

            } else{
                stack[--topIndex] = nodes[nodeIdx].right;
                stack[--topIndex] = nodes[nodeIdx].left;
                if(topIndex < 0){
                    printf_DEBUG2("Intersect stack not big enough!\n");
                    return false;
                }
            }
        }

    }
        //delete cur;
        return haveOne;
}
