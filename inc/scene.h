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
#include "camera.h"

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
        Camera* camera;

        int imageWidth;
        int imageHeight;
        int numMaterials;
        int superSampling;
        
        friend class Scene_d;

    public:
        Scene_h(): imageWidth(512), imageHeight(512), superSampling(1) {}
        Scene_h(int imageWidth, int imageHeight, int superSampling): imageWidth(imageWidth), imageHeight(imageHeight), superSampling(superSampling) {}

        void LoadObj(string filename, string mtl_basepath);
        vector<Vec3f> getImage() const {return image;}

        Scene_h& operator = (const Scene_d& deviceScene); //Copy image from the device
        void setCamera(Camera* _camera){camera = _camera; }

};

//This is the device side scene.
// 5*sizeof(pointer) in size
class Scene_d{
    public:
        int numVertices;
        int imageWidth;
        int imageHeight;
        int numTriangles;
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

        Light* light;
        Camera* camera;

        friend class Scene_h;
    public:

        Scene_d& operator = (const Scene_h& hostScene); //Copy Triangles, materials, etc to the device
    
        void computeBoundingBoxes();
        void findMinMax(Vec3f& mMin, Vec3f& mMax);

        __device__
        bool intersect(const ray& r, isect& i){ //Find the closest point of intersection
            //printf("in SCENE INTERSECT\n");
            return bvh.intersect(r, i);
        }

        __device__
        Light* getLight(){ return light; }

        __device__
        Camera* getCamera() { return camera; }

        __device__
        Vec3f ambient() const {return Vec3f(0.0,0.0,0.0);}

        ~Scene_d();

};

void AverageSuperSampling(Vec3f* smallImage, Vec3f* deviceImage, int imageWidth, int imageHeight, int superSampling);
