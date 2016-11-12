#include <stdio.h>
#include "bvh.cuh"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include "scene.h"
#include "debug.h"


__global__ void hello()
{
    printf_DEBUG("Hello world! I'm a thread in block %d\n", blockIdx.x);

}
// Expands a 10-bit integer into 30 bits
// // by inserting 2 zeros after each bit.
__device__
unsigned int expandBits(unsigned int v);

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
__device__
unsigned int morton3D(double x, double y, double z);

__device__ __inline__
Vec3d computeCentroid(const BoundingBox& BBox){
    return (BBox.getMin() + BBox.getMax()) / 2.0f;
};
__device__
int findSplit(  unsigned int* sortedMortonCodes,
        int first,
        int last);

__device__
int2 determineRange(unsigned int* sortedMortonCodes, int numTriangles, int idx);

__global__ 
void computeMortonCodesKernel(unsigned int* mortonCodes, unsigned int* object_ids, 
        BoundingBox* BBoxs, int numTriangles, Vec3d mMin, Vec3d mMax);
__global__ 
void setupLeafNodesKernel(unsigned int* sorted_object_ids, BoundingBox* BBoxs,
        LeafNode* leafNodes, int numTriangles);

__global__ 
void computeBBoxesKernel( LeafNode* leafNodes,
        InternalNode* internalNodes,
        int numTriangles);

__global__ 
void generateHierarchyKernel(unsigned int* mortonCodes,
        unsigned int* sorted_object_ids, 
        InternalNode* internalNodes,
        LeafNode* leafNodes, int numTriangles, BoundingBox* BBoxs);

void BVH_d::computeMortonCodes(Vec3d& mMin, Vec3d& mMax){
    int threadsPerBlock = 256;
    int blocksPerGrid =
        (numTriangles + threadsPerBlock - 1) / threadsPerBlock;
    computeMortonCodesKernel<<<blocksPerGrid, threadsPerBlock>>>(mortonCodes, object_ids, BBoxs, numTriangles, mMin , mMax);

}
void BVH_d::sortMortonCodes(){
    thrust::device_ptr<unsigned int> dev_mortonCodes(mortonCodes);
    thrust::device_ptr<unsigned int> dev_object_ids(object_ids);

    // Let thrust do all the work for us
    thrust::sort_by_key(dev_mortonCodes, dev_mortonCodes + numTriangles, dev_object_ids);
}

void BVH_d::setupLeafNodes(){
    int threadsPerBlock = 256;
    int blocksPerGrid =
        (numTriangles + threadsPerBlock - 1) / threadsPerBlock;
    setupLeafNodesKernel<<<blocksPerGrid, threadsPerBlock>>>(object_ids, BBoxs, leafNodes, numTriangles);

}
void BVH_d::buildTree(){
    int threadsPerBlock = 256;
    int blocksPerGrid =
        (numTriangles - 1 + threadsPerBlock - 1) / threadsPerBlock;
    setupLeafNodesKernel<<<blocksPerGrid, threadsPerBlock>>>(object_ids, BBoxs, leafNodes, numTriangles);
    cudaDeviceSynchronize();
    std::cout << "Post set up leaf nodes " << cudaGetErrorString(cudaGetLastError()) << std::endl;

    generateHierarchyKernel<<<blocksPerGrid, threadsPerBlock>>>(mortonCodes, object_ids, internalNodes , leafNodes , numTriangles, BBoxs);
    cudaDeviceSynchronize();
    std::cout << "Post Generate Hierarchy " << cudaGetErrorString(cudaGetLastError()) << std::endl;

    computeBBoxesKernel<<<blocksPerGrid, threadsPerBlock>>>(leafNodes, internalNodes, numTriangles);
    cudaDeviceSynchronize();
    std::cout << "Post compute Tree BBoxes " << cudaGetErrorString(cudaGetLastError()) << std::endl;

}

void bvh(Scene_h& scene_h)
{
    Scene_d scene_d;
    scene_d = scene_h;

    //launch the kernel
    //hello<<<NUM_BLOCKS, BLOCK_WIDTH>>>();

    //force the printf_DEBUG()s to flush
    cudaDeviceSynchronize();

    printf_DEBUG("That's all!\n");

}

//===========Begin KERNELS=============================
//===========Begin KERNELS=============================
// This kernel just computes the object id and morton code for the centroid of each bounding box
__global__ 
void computeMortonCodesKernel(unsigned int* mortonCodes, unsigned int* object_ids, 
        BoundingBox* BBoxs, int numTriangles, Vec3d mMin , Vec3d mMax){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numTriangles)
        return;

    object_ids[idx] = idx;
    Vec3d centroid = computeCentroid(BBoxs[idx]);
    centroid.x = (centroid.x - mMin.x)/(mMax.x - mMin.x);
    centroid.y = (centroid.y - mMin.y)/(mMax.y - mMin.y);
    centroid.z = (centroid.z - mMin.z)/(mMax.z - mMin.z);
    //map this centroid to unit cube
    mortonCodes[idx] = morton3D(centroid.x, centroid.y, centroid.z);
    printf_DEBUG("in computeMortonCodesKernel: idx->%d , mortonCode->%d, centroid(%0.6f,%0.6f,%0.6f)\n", idx, mortonCodes[idx], centroid.x, centroid.y, centroid.z);

};

__global__ 
void setupLeafNodesKernel(unsigned int* sorted_object_ids, 
        BoundingBox* BBoxs,
        LeafNode* leafNodes, int numTriangles){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numTriangles)
        return;
    leafNodes[idx].isLeaf = true;
    leafNodes[idx].object_id = sorted_object_ids[idx];
    leafNodes[idx].childA = nullptr;
    leafNodes[idx].childB = nullptr;
    leafNodes[idx].parent = nullptr;
    //leafNodes[idx].node_id= idx;
    leafNodes[idx].BBox= BBoxs[sorted_object_ids[idx]];
}

    __global__ 
void computeBBoxesKernel( LeafNode* leafNodes, InternalNode* internalNodes, int numTriangles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numTriangles)
        return;


    printf_DEBUG("* LEAF(%d) BB bmin(%0.6f,%0.6f,%0.6f) bmax(%0.6f,%0.6f,%0.6f) \n",idx, leafNodes[idx].BBox.bmin.x, leafNodes[idx].BBox.bmin.y, leafNodes[idx].BBox.bmin.z, leafNodes[idx].BBox.bmax.x, leafNodes[idx].BBox.bmax.y, leafNodes[idx].BBox.bmax.z);
    Node* Parent = leafNodes[idx].parent;
    while(Parent != nullptr)
    {
        if(atomicCAS(&(Parent->flag), 0 , 1))
        {
            //Parent->BBox.bEmpty = true;
            Parent->BBox.merge(Parent->childA->BBox);
            Parent->BBox.merge(Parent->childB->BBox);

            int idA = (int)((Parent->childA->isLeaf) ? (LeafNode*)Parent->childA - leafNodes: (InternalNode*)Parent->childA - internalNodes ) ;
            int idB = (int)((Parent->childB->isLeaf) ? (LeafNode*)Parent->childB - leafNodes: (InternalNode*)Parent->childB - internalNodes ) ;
            printf_DEBUG("**********parent child relationships**********\n"
                    "* parent idx (%d) bmin(%0.6f,%0.6f,%0.6f) bmax(%0.6f,%0.6f,%0.6f) \n"
                    "* childA(%d) is_leaf(%d) ob=%d bmin(%0.6f,%0.6f,%0.6f) bmax(%0.6f,%0.6f,%0.6f) \n"
                    "* childB(%d) is_leaf(%d) ob=%d bmin(%0.6f,%0.6f,%0.6f) bmax(%0.6f,%0.6f,%0.6f) \n",
                    (InternalNode*) Parent - internalNodes, Parent->BBox.bmin.x , Parent->BBox.bmin.y,Parent->BBox.bmin.z, Parent->BBox.bmax.x , Parent->BBox.bmax.y, Parent->BBox.bmax.z,
                    idA, Parent->childA->isLeaf, Parent->childA->object_id, Parent->childA->BBox.bmin.x, Parent->childA->BBox.bmin.y , Parent->childA->BBox.bmin.z, Parent->childA->BBox.bmax.x, Parent->childA->BBox.bmax.y, Parent->childA->BBox.bmax.z ,
                    idB, Parent->childB->isLeaf, Parent->childB->object_id, Parent->childB->BBox.bmin.x, Parent->childB->BBox.bmin.y , Parent->childB->BBox.bmin.z, Parent->childB->BBox.bmax.x, Parent->childB->BBox.bmax.y, Parent->childB->BBox.bmax.z );


            Parent = Parent->parent;
        }
        else{
            return;


        }



    }



}

    __global__ 
void generateHierarchyKernel(unsigned int* sortedMortonCodes,
        unsigned int* sorted_object_ids, 
        InternalNode* internalNodes,
        LeafNode* leafNodes, int numTriangles, BoundingBox* BBoxs)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx > numTriangles - 2 ) //there are n - 1 internal nodes
        return;

    internalNodes[idx].isLeaf = false ;
    internalNodes[idx].parent = nullptr ;
    internalNodes[idx].BBox = BoundingBox();
    internalNodes[idx].flag = 0;
    //internalNodes[idx].node_id = idx;

    int2 range = determineRange(sortedMortonCodes, numTriangles, idx);
    int first = range.x;
    int last = range.y;

    //Determine where to split the range.

    int split = findSplit(sortedMortonCodes, first, last);

    // Select childA.

    Node* childA;
    if (split == first)
    {
        childA = &leafNodes[split];
        //childA->BBox = BBoxs[split];
    }
    else
        childA = &internalNodes[split];

    // Select childB.

    Node* childB;
    if (split + 1 == last)
    {
        childB = &leafNodes[split + 1];
        //childB->BBox = BBoxs[split + 1];
    }
    else
        childB = &internalNodes[split + 1];

    // Record parent-child relationships.

    internalNodes[idx].childA = childA;
    internalNodes[idx].childB = childB;
    childA->parent = &internalNodes[idx];
    childB->parent = &internalNodes[idx];

}
//===========END KERNELS=============================
//===========END KERNELS=============================

    __device__
int findSplit( unsigned int* sortedMortonCodes,
        int first,
        int last)
{
    // Identical Morton codes => split the range in the middle.
    unsigned int firstCode = sortedMortonCodes[first];
    unsigned int lastCode = sortedMortonCodes[last];

    if (firstCode == lastCode)
        return (first + last) >> 1;

    // Calculate the number of highest bits that are the same
    // for all objects, using the count-leading-zeros intrinsic.

    int commonPrefix = __clz(firstCode ^ lastCode);

    // Use binary search to find where the next bit differs.
    // Specifically, we are looking for the highest object that
    // shares more than commonPrefix bits with the first one.

    int split = first; // initial guess
    int step = last - first;

    do
    {
        step = (step + 1) >> 1; // exponential decrease
        int newSplit = split + step; // proposed new position

        if (newSplit < last)
        {
            unsigned int splitCode = sortedMortonCodes[newSplit];
            int splitPrefix = __clz(firstCode ^ splitCode);
            if (splitPrefix > commonPrefix)
                split = newSplit; // accept proposal
        }
    }
    while (step > 1);

    return split;
}

    __device__
int2 determineRange(unsigned int* sortedMortonCodes, int numTriangles, int idx)
{
    //determine the range of keys covered by each internal node (as well as its children)
    //direction is found by looking at the neighboring keys ki-1 , ki , ki+1
    //the index is either the beginning of the range or the end of the range
    int direction = 0;
    int common_prefix_with_left = 0;
    int common_prefix_with_right = 0;

    common_prefix_with_right = __clz(sortedMortonCodes[idx] ^ sortedMortonCodes[idx + 1]);
    if(idx == 0){
        common_prefix_with_left = -1;
    }
    else
    {
        common_prefix_with_left = __clz(sortedMortonCodes[idx] ^ sortedMortonCodes[idx - 1]);

    }

    direction = ( (common_prefix_with_right - common_prefix_with_left) > 0 ) ? 1 : -1;
    int min_prefix_range = 0;

    if(idx == 0)
    {
        min_prefix_range = -1;

    }
    else
    {
        min_prefix_range = __clz(sortedMortonCodes[idx] ^ sortedMortonCodes[idx - direction]); 
    }

    int lmax = 2;
    int next_key = idx + lmax*direction;

    while((next_key >= 0) && (next_key <  numTriangles) && (__clz(sortedMortonCodes[idx] ^ sortedMortonCodes[next_key]) > min_prefix_range))
    {
        lmax *= 2;
        next_key = idx + lmax*direction;
    }
    //find the other end using binary search
    unsigned int l = 0;

    do
    {
        lmax = (lmax + 1) >> 1; // exponential decrease
        int new_val = idx + (l + lmax)*direction ; 

        if(new_val >= 0 && new_val < numTriangles )
        {
            unsigned int Code = sortedMortonCodes[new_val];
            int Prefix = __clz(sortedMortonCodes[idx] ^ Code);
            if (Prefix > min_prefix_range)
                l = l + lmax;
        }
    }
    while (lmax > 1);

    int j = idx + l*direction;

    int left = 0 ; 
    int right = 0;

    if(idx < j){
        left = idx;
        right = j;
    }
    else
    {
        left = j;
        right = idx;
    }

    printf_DEBUG("idx : (%d) returning range (%d, %d) \n" , idx , left, right);

    return make_int2(left,right);
}
    __device__
unsigned int expandBits(unsigned int v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
    __device__
unsigned int morton3D(double x, double y, double z)
{
    x = min(max(x * 1024.0f, 0.0f), 1023.0f);
    y = min(max(y * 1024.0f, 0.0f), 1023.0f);
    z = min(max(z * 1024.0f, 0.0f), 1023.0f);
    unsigned int xx = expandBits((unsigned int)x);
    unsigned int yy = expandBits((unsigned int)y);
    unsigned int zz = expandBits((unsigned int)z);
    return xx * 4 + yy * 2 + zz;
}
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
    //printf("t=%f\n", t);
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
    printf_DEBUG("[bmin(%f,%f,%f) bmax(%f,%f,%f)]", b.bmin.x, b.bmin.y, b.bmin.z,b.bmax.x, b.bmax.y, b.bmax.z);
}
__device__
void printNode(const Node* node, int stackDepth){
    printf("   ");
    for(int i = 0; i < stackDepth; i++)
        printf(" ");
    if(node->isALeaf())
        printf("LN: ");
    else
        printf("IN: ");
    //printf("%d id=%d ob=%d ", stackDepth, node->node_id, node->object_id);
    printBBox(node->BBox);
    //printf("\n");
}
__device__
bool BVH_d::intersect(const ray& r, isect& i) const{
#if DORECURSIVE
    i.t = 1.0e32;
    return intersect(r, i, getRoot());
#elif DOSEQ
    //printf("HERE\n");
    bool haveOne = false;
    isect* cur = new isect();
    double tmin, tmax;
    //printf("HERE\n");
    printf_DEBUG("Num Triangles %d\n", numTriangles);
    for(int j = 0; j < numTriangles; j++){
        //if(!BBoxs[object_ids[j]].intersect(r, tmin, tmax))
        //    continue;
        //if(BBoxs[object_ids[j]].intersect(r, tmin, tmax)){
        //if(BBoxs[object_ids[j]].intersect(r)){
            printf_DEBUG("%d ", object_ids[j]);
            printBBox(BBoxs[object_ids[j]]);
            printf_DEBUG(" BX");
            if(intersectTriangle(r, *cur, object_ids[j])){
                printf_DEBUG(" TX");
                if(!haveOne || (cur->t < i.t)){
                    printf_DEBUG(" t=%f", cur->t);
                    //printf("FOUND ONE t=%f\n",cur->t);
                    i = *cur;
                    haveOne = true;
                }
            }
            printf_DEBUG("\n");
        //}else{
        //    printf_DEBUG("[MISS] %d\n", object_ids[j]);

        // }
    }
    if(!haveOne) i.t = 1000.0;
    delete cur;
    //printf("Closest is %d, %f\n", i.object_id, i.t);
    return haveOne;
#elif ATTEMPT
    Node* stack[64];
    int topIndex = 64;
    stack[--topIndex] = getRoot();
    bool haveOne = false;
    isect cur;// = new isect();
    printf_DEBUG("HERE\n");
    while (topIndex != 64){
        Node* node = stack[topIndex++];
        //printNode(node, 64-topIndex);
        if(node->BBox.intersect(r)) {
            printf_DEBUG(" BX");
            if(node->isALeaf()){
                if(intersectTriangle(r, cur, ((LeafNode*)node)->object_id)){
                    printf_DEBUG(" TX");
                    if((!haveOne || (cur.t < i.t))){
                        printf_DEBUG(" t=%f", cur.t);
                        //if((!haveOne || (cur->t < i.t)) && cur->t > RAY_EPSILON){
                        i = cur;
                        haveOne = true;
                    }
                }

            } else{
                    stack[--topIndex] = node->childB;
                    stack[--topIndex] = node->childA;
                    if(topIndex < 0){
                        printf("Intersect stack not big enough!\n");
                        return false;
                    }
            }
        }

    printf_DEBUG("\n");
//    __syncthreads();
    }
    //delete cur;
    return haveOne;
#else
        bool haveOne = false;
        double tMinA;
        double tMaxA;
        double tMinB;
        double tMaxB;
        Node* childL;
        Node* childR;
        bool overlapL;
        bool overlapR;

        // Allocate traversal stack from thread-local memory,
        // and push NULL to indicate that there are no postponed nodes.
        Node* stack[64];
        //Node** stack = (Node**)malloc( 64*sizeof(Node*));
        Node** stackPtr = stack;
        *stackPtr = NULL; // push
        stackPtr++;

        // Traverse nodes starting from the root.
        Node* node = getRoot();
        if(!node->BBox.intersect(r, tMinA, tMaxA))
            return false;
        do
        {
            // Check each child node for overlap.
            childL = node->childA;
            childR = node->childB;
            overlapL = childL->BBox.intersect(r, tMinA, tMaxA);
            overlapR = childR->BBox.intersect(r, tMinB, tMaxB);

            //                                printf("%d %d\n", overlapL, overlapR);

            //                                printf("rp(%f, %f, %f), rd(%f,%f,%f) Node: %d\n", r.p.x,r.p.y,r.p.z,r.d.x,r.d.y,r.d.z, (InternalNode*)node - internalNodes);
            // Query overlaps a leaf node => check intersect
            if (overlapL && childL->isALeaf())
            {
                isect* cur = new isect();
                if(intersectTriangle(r, *cur, ((LeafNode*)childL)->object_id)){

                    if((!haveOne || (cur->t < i.t))){
                        //if((!haveOne || (cur->t < i.t)) && cur->t > RAY_EPSILON){
                        //                                            printf("LEFT FOUND ONE t=%f, N=(%f, %f, %f)f\n",cur->t, cur->N.x, cur->N.y, cur->N.z);
                        i = *cur;
                        haveOne = true;
                    }
                    }
                    delete cur;
                }

                if (overlapR && childR->isALeaf())
                {
                    isect* cur = new isect();
                    if(intersectTriangle(r, *cur, ((LeafNode*)childR)->object_id)){
                        //if((!haveOne || (cur->t < i.t)) && cur->t > RAY_EPSILON){
                        //                                            printf("RIGHT FOUND ONE t=%f, N=(%f, %f, %f)f\n",cur->t, cur->N.x, cur->N.y, cur->N.z);
                        //printf("FOUND RIGHT\n");
                        if(!haveOne || (cur->t < i.t)){
                            //printf("FOUND ONE t=%f\n",cur->t);
                            i = *cur;
                            haveOne = true;
                        }
                    }
                    delete cur;
                    }

                    // Query overlaps an internal node => traverse.
                    bool traverseL = (overlapL && !childL->isALeaf());
                    bool traverseR = (overlapR && !childR->isALeaf());

                    if (!traverseL && !traverseR)
                        node = *(--stackPtr); // pop
                    else
                    {
                        node = (traverseL) ? childL : childR;
                        if (traverseL && traverseR){
                            *stackPtr = childR; // push
                            stackPtr++;
                        }

                    }
                }
                while (node != NULL);
                if(!haveOne) i.t = 1000.0;
                //free(stack);
                //printf("Closest is %d, %f\n", i.object_id, i.t);
                return haveOne;
#endif
            }

