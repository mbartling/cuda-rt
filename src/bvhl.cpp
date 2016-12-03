
#include "bvhl.cuh"

void BVH_L::setUp(Vec3d* mvertices, Vec3d* mnormals, BoundingBox* mBBoxs, TriangleIndices* mt_indices, int mnumTriangles, Material* mmaterials, Vec3d mMin, Vec3d mMax, int* mmaterial_ids, Vec3d lightPos){
    numTriangles = mnumTriangles;
    normals = mnormals;
    vertices = mvertices;
    BBoxs = mBBoxs;
    t_indices = mt_indices;
    materials = mmaterials;
    material_ids = mmaterial_ids;

    cudaMalloc(&lightOrigin, sizeof(Vec3d));
    cudaMemcpy(lightOrigin, &lightPos, sizeof(Vec3d), cudaMemcpyHostToDevice);

    
    cudaMalloc(&nodes, (2*numTriangles - 1)*sizeof(Node_L));
    cudaMalloc(&treeBSpheres, (2*numTriangles - 1)*sizeof(BSphere));
    cudaMalloc(&sortedDims, 3*numTriangles*sizeof(double));
    cudaMalloc(&sortedObjectIds, 3*numTriangles*sizeof(int));
    cudaMalloc(&objBounds, numTriangles*sizeof(BSphere));

    // Set up for the BVH Build
    buildTree(mMin, mMax);
    cudaDeviceSynchronize();
    std::cout << "Post Build Tree " << cudaGetErrorString(cudaGetLastError()) << std::endl;

    // Build the BVH

}

//__host__ __device__ 
BVH_L::~BVH_L(){
    cudaFree(&lightOrigin);

    cudaFree(&nodes);
    cudaFree(&treeBSpheres);
    cudaFree(&sortedDims);
    cudaFree(&sortedObjectIds);
    cudaFree(&objBounds);
}
