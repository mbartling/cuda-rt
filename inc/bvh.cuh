#pragma once
#define NUM_BLOCKS 16
#define BLOCK_WIDTH 1
#include "tiny_obj_loader.h"
#include "tris.h"
#include "bbox.h"
#include "material.h"
#include "isect.h"
#include "ray.h"
#include <cuda.h>

#define DORECURSIVE 0
#define DOSEQ 0
#define ATTEMPT 1
#define WITH_CULLING 0
class Scene_d;

struct HNode{
    int parent;
    int left;
    int right;
    int next;
};

#define LEAFNODE(x)  (((x).left) == ((x).right))
#define LEAFIDX(i) ((numprims-1) + i)
#define TOLEAFIDX(i) (i - (numprims-1) )
#define NODEIDX(i) (i)

//Device BVH
class BVH_d {
    private:
        int* mortonCodes;
        unsigned int* object_ids;

        HNode* nodes;
        BoundingBox* sortedBBoxs;
        int* flags;


        // These are stored in the scene
        int numTriangles;
        Vec3d* vertices; 
        Vec3d* normals;
        BoundingBox* BBoxs;
        TriangleIndices* t_indices;
        Material* materials;
        int* material_ids;       


    public:

        void setUp(Vec3d* mvertices, Vec3d* mnormals, BoundingBox* mBBoxs, TriangleIndices* mt_indices, int mnumTriangles, Material* mmaterials, Vec3d mMin , Vec3d mMax, int* mmaterial_ids);
        ~BVH_d();
        void computeMortonCodes(Vec3d& mMin, Vec3d& mMax); //Also Generates the objectIds
        void sortMortonCodes();

        void setupLeafNodes();
        void buildTree();

        __device__
            bool intersectTriangle(const ray& r, isect&  i, int object_id) const;


        __device__
            bool intersect(const ray& r, isect& i) const;


};

class Scene_h;
void bvh(Scene_h& scene_h);

