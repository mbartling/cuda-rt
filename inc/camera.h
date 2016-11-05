#ifndef CAMERA_H
#define CAMERA_H

#include "ray.h"

class Camera
{
    public:
        __host__ __device__
        Camera();
        __host__ __device__
        void rayThrough( float x, float y, ray &r );
        __host__ __device__
        void setEye( const Vec3f &eye );
        __host__ __device__
        void setLook( float, float, float, float );
        __host__ __device__
        void setLook( const Vec3f &viewDir, const Vec3f &upDir );
        __host__ __device__
        void setFOV( float );
        __host__ __device__
        void setAspectRatio( float );

        __host__ __device__
        float getAspectRatio() { return aspectRatio; }

        __host__ __device__
        const Vec3f& getEye() const         { return eye; }
        __host__ __device__
        const Vec3f& getLook() const        { return look; }
        __host__ __device__
        const Vec3f& getU() const           { return u; }
        __host__ __device__
        const Vec3f& getV() const           { return v; }
    private:
        Mat3f m;                     // rotation matrix
        float normalizedHeight;    // dimensions of image place at unit dist from eye
        float aspectRatio;

        __host__ __device__
        void update();              // using the above three values calculate look,u,v

        Vec3f eye;
        Vec3f look;                  // direction to look
        Vec3f u,v;                   // u and v in the 
};

#endif
