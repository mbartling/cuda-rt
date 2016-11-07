#pragma once
#define NUM_BLOCKS 16
#define BLOCK_WIDTH 1
#include "tiny_obj_loader.h"
#include "tris.h"
#include "bbox.h"
#include "material.h"
#include "isect.h"
#include "ray.h"
class Scene_d;

struct Node{
    Node* childA;
    Node* childB;
    Node* parent;
    bool isLeaf;
    BoundingBox BBox;

    __device__ 
        Node() : isLeaf(false) {}
};
struct LeafNode : public Node {
    unsigned int object_id;

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
        Vec3f* vertices; 
        Vec3f* normals;
        BoundingBox* BBoxs;
        TriangleIndices* t_indices;
        Material* materials;
        int* material_ids;       


    public:

        void setUp(Vec3f* mvertices, Vec3f* mnormals, BoundingBox* mBBoxs, TriangleIndices* mt_indices, int mnumTriangles, Material* mmaterials, Vec3f mMin , Vec3f mMax, int* mmaterial_ids);
        ~BVH_d();
        void computeMortonCodes(Vec3f& mMin, Vec3f& mMax); //Also Generates the objectIds
        void sortMortonCodes();

        void setupLeafNodes();
        void buildTree();

        __device__
            bool intersectTriangle(const ray& r, isect&  i, int object_id){

                const TriangleIndices* ids = &t_indices[object_id];

                const Vec3f* a = &vertices[ids->a.vertex_index];
                const Vec3f* b = &vertices[ids->b.vertex_index];
                const Vec3f* c = &vertices[ids->c.vertex_index];

                /*
                   -DxAO = AOxD
                   AOx-D = -(-DxAO)
                   |-D AB AC| = -D*(ABxAC) = -D*normal = 1. 1x
                   |AO AB AC| = AO*(ABxAC) = AO*normal = 1. 
                   |-D AO AC| = -D*(AOxAC) = 1. 1x || AC*(-DxAO) = AC*M = 1. 1x
                   |-D AB AO| = -D*(ABxAO) = 1. 1x || (AOx-D)*AB = (DxAO)*AB = -M*AB = 1.
                   */
                float mDet;
                float mDetInv;
                float alpha;
                float beta;
                float t;
                //Moller-Trombore approach is a change of coordinates into a local uv space
                // local to the triangle
                Vec3f AB = *b - *a;
                Vec3f AC = *c - *a;

                // if (normal * -r.getDirection() < 0) return false;
                Vec3f P = r.getDirection() ^ AC;
                mDet = AB * P;
                if(fabsf(mDet) < RAY_EPSILON ) return false;

                mDetInv = 1.0f/mDet;
                Vec3f T = r.getPosition() - *a;
                alpha = T * P * mDetInv;    
                if(alpha < 0.0 || alpha > 1.0) return false;

                Vec3f Q = T ^ AB;
                beta = r.getDirection() * Q * mDetInv;
                if(beta < 0.0 || alpha + beta > 1.0) return false;
                t = AC * Q * mDetInv;

                //if( t < RAY_EPSILON ) return false;
                if(fabsf(t) < RAY_EPSILON) return false; // Jaysus this sucked
                i.bary = Vec3f(1 - (alpha + beta), alpha, beta);
                //printf("t=%f\n", t);
                if( t < 0.0 ) return false;
                i.t = t;


                // std::cout << traceUI->smShadSw() << std::endl; 
                // if(traceUI->smShadSw() && !parent->floatCheck()){
                //Smooth Shading
                i.N = (1.0 - (alpha + beta))*normals[ids->a.normal_index] + \
                      alpha*normals[ids->b.normal_index] + \
                      beta*normals[ids->c.normal_index];

                //i.N = normal;

                normalize(i.N);

                i.object_id = object_id;

                return true;


            }
            __device__
                bool intersect(const ray& r, isect& i){
                    bool haveOne = false;
                    isect* cur = new isect();
                    //printf("HERE\n");
                    for(int j = 0; j < numTriangles; j++){
                        if(intersectTriangle(r, *cur, object_ids[j])){
                            if(!haveOne || (cur->t < i.t)){
                                //printf("FOUND ONE t=%f\n",cur->t);
                                i = *cur;
                                haveOne = true;
                            }
                        }
                    }
                if(!haveOne) i.t = 1000.0;
                delete cur;
                //printf("Closest is %d, %f\n", i.object_id, i.t);
                return haveOne;

                }



            };

           class Scene_h;
           void bvh(Scene_h& scene_h);

