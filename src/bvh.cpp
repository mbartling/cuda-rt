#include "bvh.cuh"

void BVH_d::setUp(Vec3d* mvertices, Vec3d* mnormals, BoundingBox* mBBoxs, TriangleIndices* mt_indices, int mnumTriangles, Material* mmaterials, Vec3d mMin, Vec3d mMax, int* mmaterial_ids){
    numTriangles = mnumTriangles;
    normals = mnormals;
    vertices = mvertices;
    BBoxs = mBBoxs;
    t_indices = mt_indices;
    materials = mmaterials;
    material_ids = mmaterial_ids;    

    cudaMalloc(&mortonCodes, numTriangles*sizeof(unsigned int));
    cudaMalloc(&object_ids, numTriangles*sizeof(unsigned int));
    
    cudaMalloc(&leafNodes, numTriangles*sizeof(LeafNode));
    cudaMalloc(&internalNodes, (numTriangles - 1)*sizeof(InternalNode));

    // Set up for the BVH Build
    computeMortonCodes(mMin, mMax);
    cudaDeviceSynchronize();
    std::cout << "Post Compute Morton Codes " << cudaGetErrorString(cudaGetLastError()) << std::endl;
    sortMortonCodes();
    cudaDeviceSynchronize();
    std::cout << "Post Sort Morton Codes " << cudaGetErrorString(cudaGetLastError()) << std::endl;
    buildTree();
    cudaDeviceSynchronize();
    std::cout << "Post Build Tree " << cudaGetErrorString(cudaGetLastError()) << std::endl;

    // Build the BVH

}

//__host__ __device__ 
BVH_d::~BVH_d(){
    cudaFree(mortonCodes);
    cudaFree(object_ids);
    cudaFree(leafNodes);
    cudaFree(internalNodes);
}
