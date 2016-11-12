#include "scene.h"
#include <stdio.h>


// Load the OBJ and add all the triangles to a linear array
void Scene_h::LoadObj(string filename, string mtl_basepath){
    vector<shape_t> shapes;
    vector<material_t> material;
    string err;

    //Save some time by not retriangulating
    bool ret = tinyobj::LoadObj(&mAttributes, &shapes, &material, &err, filename.c_str(), mtl_basepath.c_str(), true);

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
        std::cout << "Number of faces in mesh "<< shapes[s].mesh.num_face_vertices.size() << std::endl;
        for(size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++){
        //std::cout << "Number of Vertices in face "<< (int)shapes[s].mesh.num_face_vertices[f] << std::endl;
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
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                    msg, cudaGetErrorString(__err), \
                    __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)
Scene_d& Scene_d::operator = (const Scene_h& hostScene){
    printf("Copying to Device\n");
    numVertices = hostScene.mAttributes.vertices.size()/3;
    int numNormals = hostScene.mAttributes.normals.size()/3;
    printf("num of vertices : %d \n" , numVertices);
    numTriangles = hostScene.t_indices.size();
    printf("num of triangles : %d \n" , numTriangles);
    numMaterials = hostScene.materials.size();
    printf("num of materials : %d \n", numMaterials);
    imageWidth = hostScene.imageWidth * hostScene.superSampling;
    imageHeight = hostScene.imageHeight * hostScene.superSampling;
    printf("Device imageWidth=%d, imageHeight=%d\n", imageWidth, imageHeight);

    //Allocate Space for everything
    cudaMalloc(&vertices, numVertices*sizeof(Vec3d));
    cudaCheckErrors("cudaMalloc vertices fail");
    cudaMalloc(&normals, numNormals*sizeof(Vec3d));
    cudaCheckErrors("cudaMalloc normals fail");

    cudaMalloc(&BBoxs, numTriangles*sizeof(BoundingBox));
    cudaCheckErrors("cudaMalloc BBoxs fail");
    cudaMalloc(&t_indices, numTriangles*sizeof(TriangleIndices));
    cudaCheckErrors("cudaMalloc triangle indices fail");
    cudaMalloc(&materials, numMaterials*sizeof(Material));
    cudaCheckErrors("cudaMalloc materials fail");
    cudaMalloc(&material_ids, numTriangles*sizeof(int));
    cudaCheckErrors("cudaMalloc material ids fail");

    cudaMalloc(&image, imageWidth*imageHeight*sizeof(Vec3d));
    cudaCheckErrors("cudaMalloc image fail");
    cudaMalloc(&camera, sizeof(Camera));
    cudaCheckErrors("cudaMalloc Camera fail");
    cudaMalloc(&seeds, imageWidth*imageHeight*sizeof(uint32_t));
    cudaCheckErrors("cudaMalloc seeds fail");
    
    cudaDeviceSynchronize();

    //Copy stuff
    //Vec3d* hverts = (Vec3d*)malloc(numVertices*sizeof(Vec3d));
    //for(int i = 0; i < numVertices; i++)
    //    hverts[i] = hostScene.mAttributes.vertices[i];
    cudaMemcpy(vertices, hostScene.mAttributes.vertices.data(), numVertices*sizeof(Vec3d), cudaMemcpyHostToDevice);
    //cudaMemcpy(vertices, hverts, numVertices*sizeof(Vec3d), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy vertices fail");
    printf("HERE\n");
    //for(int i = 0; i < numVertices; i++)
    //    hverts[i] = hostScene.mAttributes.normals[i];
    //cudaMemcpy(normals, hverts, numVertices*sizeof(Vec3d), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy normals fail");
    std::cout << hostScene.mAttributes.normals.size() << endl;
    cudaMemcpy(normals, hostScene.mAttributes.normals.data(), numNormals*sizeof(Vec3d), cudaMemcpyHostToDevice);

    //free(hverts);
    cudaMemcpy(t_indices, hostScene.t_indices.data(), numTriangles*sizeof(TriangleIndices), cudaMemcpyHostToDevice);
    cudaMemcpy(material_ids, hostScene.material_ids.data(), numTriangles*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(materials, hostScene.materials.data(), numMaterials*sizeof(Material), cudaMemcpyHostToDevice);
    cudaMemcpy(camera, hostScene.camera, sizeof(Camera), cudaMemcpyHostToDevice);

    std::cout << "Done Copying basic objects to device" << std::endl;

    computeBoundingBoxes();
    cudaDeviceSynchronize();
    std::cout << "Post Scene BBoxes " << cudaGetErrorString(cudaGetLastError()) << std::endl;



    Vec3d mMin;
    Vec3d mMax;
    findMinMax(mMin, mMax);
    cudaDeviceSynchronize();
    std::cout << "Post Find Min Max " << cudaGetErrorString(cudaGetLastError()) << std::endl;
    bvh.setUp(vertices,normals, BBoxs, t_indices, numTriangles, materials, mMin , mMax, material_ids);
    cudaDeviceSynchronize();
    std::cout << "Post BVH Setup " << cudaGetErrorString(cudaGetLastError()) << std::endl;
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

