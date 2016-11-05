#pragma once
#include "scene.h"
#include "bitmap.h"
#include "Light.h"

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
        }
        
        RayTracer(int imageWidth, int imageHeight, int superSampling): imageWidth(512), imageHeight(512), superSampling(1), hostScene(imageWidth, imageHeight, superSampling), depth(1)
        {
                image = new RGB[imageWidth*imageHeight];
        }

        void LoadObj(string filename){ hostScene.LoadObj(filename); }
        void setUpDevice(){ deviceScene = hostScene; }
        void setHostLight(const Light_h& mLight) { hostLight = mLight; }
        
        void pullRaytracedImage(){ 
            hostScene = deviceScene; 
            vector<Vec3f> hostSceneImage = hostScene.getImage();
            for(int i = 0; i < hostSceneImage.size(); i++){
                image[i].r = (unsigned char)(255.0 * hostSceneImage[i].x);
                image[i].g = (unsigned char)(255.0 * hostSceneImage[i].y);
                image[i].b = (unsigned char)(255.0 * hostSceneImage[i].z);
            }

        }
        void writeImage(string filename) { 
            
            writeBMP(filename.c_str(), imageWidth, imageHeight, (unsigned char*)image); 
        }

        void run();
        

        ~RayTracer(){ delete[] image; }


};
