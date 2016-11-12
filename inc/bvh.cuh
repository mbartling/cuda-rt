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

struct Node{
    Node* childA;
    Node* childB;
    Node* parent;
    int flag;
    bool isLeaf;
    BoundingBox BBox;
    unsigned int object_id;
    //unsigned int node_id; //debugging

    __device__ 
        Node() : isLeaf(false) , flag(0), parent(nullptr) {}
    __device__
        bool isALeaf() const {return isLeaf;}
};
struct LeafNode : public Node {

    __device__
        LeafNode() {
            this->isLeaf = true;
        }
};
struct InternalNode : public Node {
    __device__
        InternalNode() {
            this->isLeaf = false;
        }
};
//Device BVH
class BVH_d {
    private:
        unsigned int* mortonCodes;
        unsigned int* object_ids;

        LeafNode*       leafNodes; //numTriangles
        InternalNode*   internalNodes; //numTriangles - 1

        // These are stored in the scene
        int numTriangles;
        Vec3d* vertices; 
        Vec3d* normals;
        BoundingBox* BBoxs;
        TriangleIndices* t_indices;
        Material* materials;
        int* material_ids;       

        __device__
            Node* getRoot() const { return &internalNodes[0];}

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
                bool intersect(const ray& r, isect& i, Node* node) const {
                    bool haveOne = false;
                    double tMin;
                    double tMax;

                    if(!node->BBox.intersect(r, tMin, tMax))
                        return false;

                    if(node->isLeaf){
                        isect* cur = new isect();
                        if(intersectTriangle(r, *cur, ((LeafNode*)node)->object_id)){
                           // if(cur->t < i.t && cur->t > RAY_EPSILON){
                            if(cur->t < i.t){
                                i = *cur;
                                haveOne = true;
                            }

                        } 

                        delete cur;
                        return haveOne;

                        } //if leaf
                        else{
                            // Sanity
                            return intersect(r, i, node->childA) || intersect(r, i, node->childB);
                        }


                    }
                    __device__
                        bool intersect(const ray& r, isect& i) const;


                };

            class Scene_h;
            void bvh(Scene_h& scene_h);

