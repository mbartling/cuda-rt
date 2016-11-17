#include "vec.h"


class BSphere {
    public:
        Vec3d  pos;
        double radius;
        bool isEmpty;

        __host__ __device__
        BSphere() : isEmpty(true) {}
        __host__ __device__
        BSphere(const Vec3d& a, const Vec3d& b) : isEmpty(false) {
            pos = (b + a)/2;
            radius = norm(b-a)/2;
        
        }


};
/*
class BSphereProj {
    public:
    Vec3d pos; //Project onto principle axis (remove biggest component)
    double radius;
    bool isEmpty;
    
    __host__ __device__
    BSphereProj() : isEmpty(true){}
    BSphereProj(Vec3d lightOrigin, double projRadius, const BSphere& bs){
        double theta = atan2(bs.radius/norm(bs.pos - lightOrigin);
        double Area = 2*M_PI*(1 - cos(theta));
        radius = projRadius*sin(theta):
        pos = ((bs.pos - lightOrigin)/norm(bs.pos - lightOrigin))*projRadius*cos(theta);

    } 

};
*/
double ProjectionArea(Vec3d lightOrigin, double projRadius, const BSphere& bs){
        double theta = atan2(bs.radius/norm(bs.pos - lightOrigin);
        return 2*M_PI*(1 - cos(theta));
}
