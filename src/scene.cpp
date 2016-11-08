#include "scene.h"
#include <stdio.h>


// Load the OBJ and add all the triangles to a linear array
void Scene_h::LoadObj(string filename, string mtl_basepath){
    vector<shape_t> shapes;
    vector<material_t> material;
    string err;

    bool ret = tinyobj::LoadObj(&mAttributes, &shapes, &material, &err, filename.c_str(), mtl_basepath.c_str());

    if (!err.empty()) { // `err` may contain warning message.
        std::cerr << err << std::endl;
    }

    if (!ret) {
        exit(1);
    }

    for(size_t m = 0; m < material.size(); m++){
        Material mat;
        mat.ka = Vec3d(material[m].ambient);
        mat.kd = Vec3d(material[m].diffuse);
        mat.ks = Vec3d(material[m].specular);
        mat.kt = Vec3d(material[m].transmittance);
        mat.ke = Vec3d(material[m].emission);
        mat.kr = Vec3d(material[m].specular); //Hack
        mat.shininess = material[m].shininess;
        mat.ior = material[m].ior;
        mat.dissolve = material[m].dissolve;
        mat.setBools();

        this->materials.push_back(mat);
    
    }
    //For each shape
    for(size_t s = 0; s < shapes.size(); s++){

        //For each Triangle Add the 
        size_t index_offset = 0;
        for(size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++){
            TriangleIndices index;
            index.a = shapes[s].mesh.indices[index_offset + 0];
            index.b = shapes[s].mesh.indices[index_offset + 1];
            index.c = shapes[s].mesh.indices[index_offset + 2];

            t_indices.push_back(index);
            material_ids.push_back(shapes[s].mesh.material_ids[f]);
            index_offset += 3;
        }
        

    }
}

Scene_h& Scene_h::operator = (const Scene_d& deviceScene){
    Vec3d* smallImage;
    cudaMalloc(&smallImage, imageWidth*imageHeight*sizeof(Vec3d));

    AverageSuperSampling(smallImage, deviceScene.image, imageWidth, imageHeight, superSampling);
    image.resize(imageWidth*imageHeight);
    
    cudaMemcpy(image.data(), smallImage, imageWidth*imageHeight*sizeof(Vec3d), cudaMemcpyDeviceToHost);
    
    cudaFree(smallImage);
}

Scene_d& Scene_d::operator = (const Scene_h& hostScene){
    numVertices = hostScene.mAttributes.vertices.size();
    numTriangles = hostScene.t_indices.size();
    printf("num of triangles : %d " , numTriangles);
    numMaterials = hostScene.materials.size();
    imageWidth = hostScene.imageWidth * hostScene.superSampling;
    imageHeight = hostScene.imageHeight * hostScene.superSampling;

    //Allocate Space for everything
    cudaMalloc(&vertices, numVertices*sizeof(Vec3d));
    cudaMalloc(&normals, numVertices*sizeof(Vec3d));

    cudaMalloc(&BBoxs, numTriangles*sizeof(BoundingBox));
    cudaMalloc(&t_indices, numTriangles*sizeof(TriangleIndices));
    cudaMalloc(&materials, numMaterials*sizeof(Material));
    cudaMalloc(&material_ids, numTriangles*sizeof(int));

    cudaMalloc(&image, imageWidth*imageHeight*sizeof(Vec3d));
    cudaMalloc(&camera, sizeof(Camera));
    cudaMalloc(&seeds, imageWidth*imageHeight*sizeof(uint32_t));

    //Copy stuff
    cudaMemcpy(vertices, hostScene.mAttributes.vertices.data(), numVertices*sizeof(Vec3d), cudaMemcpyHostToDevice);
    cudaMemcpy(normals, hostScene.mAttributes.normals.data(), numVertices*sizeof(Vec3d), cudaMemcpyHostToDevice);
    cudaMemcpy(t_indices, hostScene.t_indices.data(), numTriangles*sizeof(TriangleIndices), cudaMemcpyHostToDevice);
    cudaMemcpy(material_ids, hostScene.material_ids.data(), numTriangles*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(materials, hostScene.materials.data(), numMaterials*sizeof(Material), cudaMemcpyHostToDevice);
    cudaMemcpy(camera, hostScene.camera, sizeof(Camera), cudaMemcpyHostToDevice);

    computeBoundingBoxes();
    


    Vec3d mMin;
    Vec3d mMax;
    findMinMax(mMin, mMax);
    bvh.setUp(vertices,normals, BBoxs, t_indices, numTriangles, materials, mMin , mMax, material_ids);
    printf("found min(%0.6f, %0.6f , %0.6f)" , mMin.x , mMin.y , mMin.z);
    printf("found max(%0.6f, %0.6f , %0.6f)" , mMax.x , mMax.y , mMax.z);


    return *this;
}

Scene_d::~Scene_d(){
    cudaFree(vertices);
    cudaFree(normals);
    cudaFree(BBoxs);
    cudaFree(t_indices);
    cudaFree(image);
    cudaFree(camera);
    cudaFree(seeds);
}

