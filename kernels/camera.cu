#include "camera.h"
#include "ray.h"

#define PI 3.14159265359
#define SHOW(x) (cerr << #x << " = " << (x) << "\n")

using namespace std;

__host__ __device__
Camera::Camera(){
    aspectRatio = 1;
    normalizedHeight = 1;

    eye = Vec3d(0.f,0.f,0.f);
    u = Vec3d(1.f,0.f,0.f);
    v = Vec3d(0.f,1.f,0.f);
    look = Vec3d(0.f,0.f,-1.f);
    aperature = 1;
    fov  = 30;
}

__host__ __device__
    void
Camera::rayThrough(double x, double y, ray &r)
    // Ray through normalized window point x,y.  In normalized coordinates
    // the camera's x and y vary both vary from 0 to 1.
{
    x -= 0.5f;
    y -= 0.5f;
    Vec3d dir = look + x*u + y*v;
    normalize(dir);
    r.p = eye;
    r.d = dir;
}

__host__ __device__
    void
Camera::setEye(const Vec3d &eye)
{
    this->eye = eye;
}

__host__ __device__
void 
Camera::setAperature(double a){ aperature = a; }

__host__ __device__
    void
Camera::setLook(double r, double i, double j, double k)
    // Set the direction for the camera to look using a quaternion.  The
    // default camera looks down the neg z axis with the pos y axis as up.
    // We derive the new look direction by rotating the camera by the
    // quaternion rijk.
{
    //set look matrix
    m.x.x = 1.f - 2.f * (i*i + j*j);
    m.x.y = 2.0 * (r*i - j*k);
    m.x.z = 2.f * (j*r + i*k);

    m.y.x = 2.f * (r*i + j*k);
    m.y.y = 1.f - 2.f * (j*j + r*r) ;
    m.y.z = 2.f * (i*j - r*k);

    m.z.x = 2.f * (j*r - i*k);
    m.z.y = 2.f * (i*j + r*k);
    m.z.z = 1.f - 2.f * (i*i + r*r);

    update();
}

__host__ __device__
    void
Camera::setLook(const Vec3d &viewDir, const Vec3d &upDir)
{
    Vec3d z = -viewDir;
    const Vec3d &y = upDir;
    Vec3d x = y ^ z;

    //m = Mat3d(x.x, x.y, x.z, y.x, y.y, y.z, z.x, z.y, z.z).transpose();
    m = Mat3d(x.x, y.x, z.x, x.y, y.y, z.y, x.z, y.z, z.z);
    update();
}

__host__ __device__
    void
Camera::setFOV( double fov )
    // fov - field of view (height) in degrees
{
    fov /= (180.f / PI); // convert to radians
    normalizedHeight = 2.f * tan(fov/2.f);
    update();
}

__host__ __device__
    void
Camera::setAspectRatio(double ar)
    // ar - ratio of width to height
{
    aspectRatio = ar;
    update();
}

__host__ __device__
    void
Camera::update()
{
    u = m * Vec3d(1.f, 0.f, 0.f) * normalizedHeight*aspectRatio;
    v = m * Vec3d(0.f, 1.f, 0.f) * normalizedHeight;
    look = m * Vec3d(0.f, 0.f, -1.f);
}
