#pragma once
#include "vec.h"

#include "tiny_obj_loader.h"
#include "material.h"
#include "isect.h"
#include "ray.h"
#include "bsphere.cuh"
#include "tris.h"
#include "bbox.h"
#include <cuda.h>

class Scene_d;

struct Node_L{
    int parent;
    int childA;
    int childB;
    //Indexes into the sortedDims and sortedObjectIds
    //0: radius, 1: theta, 2: phi
    int dmins[3];
    int dmaxs[3];
};


#define NODE_IS_LEAF(x) (x.childA == x.childB)
#define MY_IDX(idx) ((1<<depth) - 1 + idx)
#define FIND_CHILDA_OFFSET(idx) ((1 << (depth+1)) - 1 + 2*idx)

class BVH_L{
    public:
        Node_L* nodes; //2*numTriangles - 1
        double*  sortedDims;
        int*    sortedObjectIds;

        BSphere* treeBSpheres;
        BSphere* objBounds;

        // These are stored in the scene
        int numTriangles;
        Vec3d* vertices; 
        Vec3d* normals;
        BoundingBox* BBoxs;
        TriangleIndices* t_indices;
        Material* materials;
        int* material_ids;       

        Vec3d* lightOrigin;

        // Interface
        void setUp(Vec3d* mvertices, Vec3d* mnormals, BoundingBox* mBBoxs, TriangleIndices* mt_indices, int mnumTriangles, Material* mmaterials, Vec3d mMin , Vec3d mMax, int* mmaterial_ids, Vec3d lightPos);
        ~BVH_L();
        
        void buildTree(Vec3d mMin, Vec3d mMax);
        void sortDimensions();

        __device__
            bool intersectTriangle(const ray& r, isect&  i, int object_id) const;


        __device__
            bool intersect(const ray& r, isect& i) const {}
        __device__
            bool intersectAny(const ray& r, isect& i) const;

};
