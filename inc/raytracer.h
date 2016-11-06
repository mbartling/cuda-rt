#pragma once
#include "scene.h"
#include "bitmap.h"
#include "Light.h"
#include <stdio.h>

struct RGB {
    unsigned char r;
    unsigned char g;
    unsigned char b;
};
class RayTracer{
    private:
        int imageWidth;
        int imageHeight;
        int superSampling; //1x 2x 4x 16x 

        int depth;

        RGB* image; //Host side
        Scene_h hostScene;
        Scene_d deviceScene;
        Light_h hostLight;
    
    public:

        RayTracer(): imageWidth(512), imageHeight(512), superSampling(1), hostScene(imageWidth, imageHeight, superSampling), depth(1)
        {
                image = new RGB[imageWidth*imageHeight];
                hostLight.position = Vec3f(10.0, 10.0, 10.0);
                hostLight.orientation = Vec3f(10.0, 10.0, 10.0);
                normalize(hostLight.orientation);
                hostLight.color = Vec3f(1.0, 1.0, 1.0);
        }
        
        RayTracer(int imageWidth, int imageHeight, int superSampling): imageWidth(512), imageHeight(512), superSampling(1), hostScene(imageWidth, imageHeight, superSampling), depth(1)
        {
                image = new RGB[imageWidth*imageHeight];
                hostLight.position = Vec3f(10.0, 10.0, 10.0);
                hostLight.orientation = Vec3f(10.0, 10.0, 10.0);
                normalize(hostLight.orientation);
                hostLight.color = Vec3f(1.0, 1.0, 1.0);
        }

        void LoadObj(string filename, string mtl_basepath){ hostScene.LoadObj(filename, mtl_basepath); }
        void setUpDevice(){ deviceScene = hostScene; }
        void setHostLight(const Light_h& mLight) { 
            hostLight = mLight; 
            normalize(hostLight.orientation);
        }
        
        void pullRaytracedImage(){ 
            hostScene = deviceScene; 
            vector<Vec3f> hostSceneImage = hostScene.getImage();
            printf("imageSize %d\n", hostSceneImage.size());
            for(int i = 0; i < hostSceneImage.size(); i++){
                image[i].r = (unsigned char)(255.0 * hostSceneImage[i].x);
                image[i].g = (unsigned char)(255.0 * hostSceneImage[i].y);
                image[i].b = (unsigned char)(255.0 * hostSceneImage[i].z);
                //printf("%d %d %d\n", image[i].r, image[i].g, image[i].b);
            }

        }
        void writeImage(string filename) { 
            
            writeBMP(filename.c_str(), imageWidth, imageHeight, (unsigned char*)image); 
        }

        void run();
        

        ~RayTracer(){ delete[] image; }


};
