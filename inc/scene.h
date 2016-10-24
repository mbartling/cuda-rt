#pragma once
#include "vec.h"
#include "tiny_obj_loader.h"
#include <vector>
#include <string>
#include "tris.h"
#include "bbox.h"
#include "bvh.h"
#include <stdlib.h>
#include "material.h"
#include "isect.h"

using namespace tinyobj;
using std::vector;
using std::cout;
using std::endl;
using std::string;



//Forward Declarations
class Scene_d;
class Scene_h;

// This is a host side scene class
// It holds all of the triangles/mesh_t from the obj file
// We can copy a scene to and from the host/device using copy assignment
class Scene_h{
    private:
        // Host Side
        attrib_t mAttributes;
        vector<Vec3f> image;
        vector<TriangleIndices> t_indices;
        vector<int> material_ids;
        vector<Material> materials;

        int imageWidth;
        int imageHeight;
        int numMaterials;
        int superSampling;
        
        friend class Scene_d;

    public:
        Scene_h(): imageWidth(512), imageHeight(512), superSampling(1) {}
        Scene_h(int imageWidth, int imageHeight, int superSampling): imageWidth(imageWidth), imageHeight(imageHeight), superSampling(superSampling) {}

        void LoadObj(string filename);

        Scene_h& operator = (const Scene_d& deviceScene); //Copy image from the device


};

//This is the device side scene.
// 5*sizeof(pointer) in size
class Scene_d{
    public:
        int numVertices;
        int numTriangles;
        int imageWidth;
        int imageHeight;
        int numMaterials;

        Vec3f* vertices;
        Vec3f* normals;
        Vec3f* texcoords;

        Material* materials;
        int* material_ids;

        BoundingBox* BBoxs; //Per Triangle Bounding Box

        TriangleIndices* t_indices;

        Vec3f* image;
        BVH_d bvh;

        friend class Scene_h;
    public:

        Scene_d& operator = (const Scene_h& hostScene); //Copy Triangles, materials, etc to the device
    
        void computeBoundingBoxes();
        void findMinMax(Vec3f& mMin, Vec3f& mMax);

        //This should be in BVH, this is temporary
        __device__
            bool intersectTriangle(const ray& r, isect&  i, int object_id){

                TriangleIndices ids = t_indices[object_id];

                Vec3f a = vertices[ids.a.vertex_index];
                Vec3f b = vertices[ids.b.vertex_index];
                Vec3f c = vertices[ids.c.vertex_index];

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
                Vec3f rDir = r.getDirection();
                //Moller-Trombore approach is a change of coordinates into a local uv space
                // local to the triangle
                Vec3f AB = b - a;
                Vec3f AC = c - a;

                // if (normal * -r.getDirection() < 0) return false;
                Vec3f P = rDir ^ AC;
                mDet = AB * P;
                if(fabsf(mDet) < RAY_EPSILON ) return false;

                mDetInv = 1/mDet;
                Vec3f T = r.getPosition() - a;
                alpha = T * P * mDetInv;    
                if(alpha < 0 || alpha > 1) return false;

                Vec3f Q = T ^ AB;
                beta = rDir * Q * mDetInv;
                if(beta < 0 || alpha + beta > 1) return false;
                t = AC * Q * mDetInv;

                if(fabsf(t) < RAY_EPSILON) return false; // Jaysus this sucked
                i.bary = Vec3f(1 - (alpha + beta), alpha, beta);
                i.t = t;


                // std::cout << traceUI->smShadSw() << std::endl; 
                // if(traceUI->smShadSw() && !parent->floatCheck()){
                //Smooth Shading
                Vec3f aN = normals[ids.a.normal_index];
                Vec3f bN = normals[ids.b.normal_index];
                Vec3f cN = normals[ids.c.normal_index];
                i.N = (1 - (alpha + beta))*aN + \
                      alpha*bN + \
                      beta*cN;

                //i.N = normal;

                normalize(i.N);

                i.object_id = object_id;
                //if(!parent->materials.empty() && parent->hasVertexMaterials()){
                Material aM;
                //TODO Be able to uncomment the following lines
                //int material_id = material_ids[object_id];
                //aM += (1 - (alpha + beta))*(materials[ids.a]); 
                //aM +=                alpha*(materials[ids.b]); 
                //aM +=                beta* (materials[ids.c]); 

                i.material = aM;

                return true;


            }
        __device__
        bool intersect(const ray& r, isect& i){ //Find the closest point of intersection
            // Begin Linear Search, remove when BVH works
            isect curr;
            bool haveOne = false;
            for(int object_id = 0; object_id < numTriangles; object_id++){
                if(intersectTriangle(r, curr, object_id)){
                    if(!haveOne || (curr.t < i.t)){
                        i = curr;
                        haveOne = true;
                    }
            }

            return haveOne;
            // End Linear Search

            //Uncomment when the BVH is done
//            return bvh.intersect(r, i);
        }

        ~Scene_d();

};

void AverageSuperSampling(Vec3f* smallImage, Vec3f* deviceImage, int imageWidth, int imageHeight, int superSampling);
