#ifndef CAMERA_H
#define CAMERA_H

#include "ray.h"

class Camera
{
    public:
        __host__ __device__
        Camera();
        __host__ __device__
        void rayThrough( double x, double y, ray &r );
        __host__ __device__
        void setEye( const Vec3d &eye );
        __host__ __device__
        void setLook( double, double, double, double );
        __host__ __device__
        void setLook( const Vec3d &viewDir, const Vec3d &upDir );
        __host__ __device__
        void setFOV( double );
        __host__ __device__
        void setAspectRatio( double );

        __host__ __device__
        double getAspectRatio() { return aspectRatio; }

        __host__ __device__
        const Vec3d& getEye() const         { return eye; }
        __host__ __device__
        const Vec3d& getLook() const        { return look; }
        __host__ __device__
        const Vec3d& getU() const           { return u; }
        __host__ __device__
        const Vec3d& getV() const           { return v; }
    private:
        Mat3d m;                     // rotation matrix
        double normalizedHeight;    // dimensions of image place at unit dist from eye
        double aspectRatio;

        __host__ __device__
        void update();              // using the above three values calculate look,u,v

        Vec3d eye;
        Vec3d look;                  // direction to look
        Vec3d u,v;                   // u and v in the 
};

#endif
