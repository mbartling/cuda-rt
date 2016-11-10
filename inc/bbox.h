#pragma once
#include "common.h"
#include "ray.h"
#include "vec.h"
#define USENEW 0

class BoundingBox {
    public:

    bool bEmpty;
    bool dirty;
    Vec3d bmin;
    Vec3d bmax;
    double bArea;
    double bVolume;

    public:

     __host__ __device__
        BoundingBox() : bEmpty(true) {}

    __host__ __device__
        BoundingBox(Vec3d bMin, Vec3d bMax) : bmin(bMin), bmax(bMax), bEmpty(false), dirty(true) {}

    __host__ __device__ __inline__
        Vec3d getMin() const { return bmin; }
    __host__ __device__ __inline__
        Vec3d getMax() const { return bmax; }
    __host__ __device__
        bool isEmpty() { return bEmpty; }

    __host__ __device__
        void setMin(Vec3d bMin) {
            bmin = bMin;
            dirty = true;
            bEmpty = false;
        }
    __host__ __device__
        void setMax(Vec3d bMax) {
            bmax = bMax;
            dirty = true;
            bEmpty = false;
        }
    __host__ __device__
        void setMin(int i, double val) {
            if (i == 0) { bmin.x = val; dirty = true; bEmpty = false; }
            else if (i == 1) { bmin.y = val; dirty = true; bEmpty = false; }
            else if (i == 2) { bmin.z = val; dirty = true; bEmpty = false; }
        }
    __host__ __device__
        void setMax(int i, double val) {
            if (i == 0) { bmax.x = val; dirty = true; bEmpty = false; }
            else if (i == 1) { bmax.y = val; dirty = true; bEmpty = false; }
            else if (i == 2) { bmax.z = val; dirty = true; bEmpty = false; }
        }
    __host__ __device__
        void setEmpty() {
            bEmpty = true;
        }

    // Does this bounding box intersect the target?
    __host__ __device__
        bool intersects(const BoundingBox &target) const {
            return ((target.getMin().x - RAY_EPSILON <= bmax.x) && (target.getMax().x + RAY_EPSILON >= bmin.x) &&
                    (target.getMin().y - RAY_EPSILON <= bmax.y) && (target.getMax().y + RAY_EPSILON >= bmin.y) &&
                    (target.getMin().z - RAY_EPSILON <= bmax.z) && (target.getMax().z + RAY_EPSILON >= bmin.z));
        }

    // does the box contain this point?
    __host__ __device__
        bool intersects(const Vec3d& point) const {
            return ((point.x + RAY_EPSILON >= bmin.x) && (point.y + RAY_EPSILON >= bmin.y) && (point.z + RAY_EPSILON >= bmin.z) &&
                    (point.x - RAY_EPSILON <= bmax.x) && (point.y - RAY_EPSILON <= bmax.y) && (point.z - RAY_EPSILON <= bmax.z));
        }

    // if the ray hits the box, put the "t" value of the intersection
    // closest to the origin in tMin and the "t" value of the far intersection
    // in tMax and return true, else return false.
    // Using Kay/Kajiya algorithm.
    __host__ __device__
        bool intersect(const ray& r, double& tMin, double& tMax) const {
#if USENEW
            Vec3d tmp = r.d;
//            if (fabs(tmp.x) < RAY_EPSILON)
//                tmp.x = (tmp.x < 0) ? -0 : 0;
//            if (fabs(tmp.y) < RAY_EPSILON)
//                tmp.y = (tmp.y < 0) ? -0 : 0;
//            if (fabs(tmp.z) < RAY_EPSILON)
//                tmp.z = (tmp.z < 0) ? -0 : 0;

            Vec3d invdir = recip(tmp);
            double tymin;
            double tymax;
            double tzmin;
            double tzmax;

            if (invdir.x >= 0){
                tMin = (bmin.x - r.p.x) * invdir.x;
                tMax = (bmax.x - r.p.x) * invdir.x;
            } else {
                tMax = (bmin.x - r.p.x) * invdir.x;
                tMin = (bmax.x - r.p.x) * invdir.x;
            }
            if (invdir.y >= 0){
                tymin = (bmin.y - r.p.y) * invdir.y;
                tymax = (bmax.y - r.p.y) * invdir.y;
            } else {
                tymax = (bmin.y - r.p.y) * invdir.y;
                tymin = (bmax.y - r.p.y) * invdir.y;
            }
//            if( (tMin > tymax) || (tymin > tMax))
//                return false;
            if (tymin > tMin)
                tMin = tymin;
            if (tymax > tMax)
                tMax = tymax;
            
            if (invdir.z >= 0){
                tzmin = (bmin.z - r.p.z) * invdir.z;
                tzmax = (bmax.z - r.p.z) * invdir.z;
            } else {
                tzmax = (bmin.z - r.p.z) * invdir.z;
                tzmin = (bmax.z - r.p.z) * invdir.z;
            }

            if( (tMin > tzmax) || (tzmin > tMax))
                return false;
            if (tzmin > tMin)
                tMin = tzmin;
            if (tzmax < tMax)
                tMax = tzmax;
            //return ((tMin < 1000.0) && (tMax > RAY_EPSILON));
            return true;

#else
            //Vec3d R0 = r.getPosition();
            //Vec3d Rd = r.getDirection();
            tMin = -1.0e307; // 1.0e308 is close to infinity... close enough for us!
            tMax = 1.0e307;
            double ttemp;

            double vd = r.d.x;
            double v1, v2, t1, t2;
            // if the ray is parallel to the face's plane (=0.0)
            if( vd != 0.0 ){
                v1 = bmin.x - r.p.x;
                v2 = bmax.x - r.p.x;
                // two slab intersections
                t1 = v1/vd;
                t2 = v2/vd;
                if ( t1 > t2 ) { // swap t1 & t2
                    ttemp = t1;
                    t1 = t2;
                    t2 = ttemp;
                }
                if (t1 > tMin) tMin = t1;
                if (t2 < tMax) tMax = t2;
                if (tMin > tMax) return false; // box is missed
                if (tMax < RAY_EPSILON) return false; // box is behind ray
            }
            vd = r.d.y;
            // if the ray is parallel to the face's plane (=0.0)
            if( vd != 0.0 ){
                v1 = bmin.y - r.p.y;
                v2 = bmax.y - r.p.y;
                // two slab intersections
                t1 = v1/vd;
                t2 = v2/vd;
                if ( t1 > t2 ) { // swap t1 & t2
                    ttemp = t1;
                    t1 = t2;
                    t2 = ttemp;
                }
                if (t1 > tMin) tMin = t1;
                if (t2 < tMax) tMax = t2;
                if (tMin > tMax) return false; // box is missed
                if (tMax < RAY_EPSILON) return false; // box is behind ray
            }
            vd = r.d.z;
            // if the ray is parallel to the face's plane (=0.0)
            if( vd != 0.0 ){
                v1 = bmin.z - r.p.z;
                v2 = bmax.z - r.p.z;
                // two slab intersections
                t1 = v1/vd;
                t2 = v2/vd;
                if ( t1 > t2 ) { // swap t1 & t2
                    ttemp = t1;
                    t1 = t2;
                    t2 = ttemp;
                }
                if (t1 > tMin) tMin = t1;
                if (t2 < tMax) tMax = t2;
                if (tMin > tMax) return false; // box is missed
                if (tMax < RAY_EPSILON) return false; // box is behind ray
            }
            return true; // it made it past all 3 axes.
#endif
        }

    __host__ __device__
        void operator=(const BoundingBox& target) {
            bmin = target.bmin;
            bmax = target.bmax;
            bArea = target.bArea;
            bVolume = target.bVolume;
            dirty = target.dirty;
            bEmpty = target.bEmpty;
        }

    __host__ __device__
        double area() {
            if (bEmpty) return 0.0;
            else if (dirty) {
                bArea = 2.0 * ((bmax.x - bmin.x) * (bmax.y - bmin.y) + (bmax.y - bmin.y) * (bmax.z - bmin.z) + (bmax.z - bmin.z) * (bmax.x - bmin.x));
                dirty = false;
            }
            return bArea;
        }

    __host__ __device__
        double volume() {
            if (bEmpty) return 0.0;
            else if (dirty) {
                bVolume = ((bmax.x - bmin.x) * (bmax.y - bmin.y) * (bmax.z - bmin.z));
                dirty = false;
            }
            return bVolume;
        }

    __host__ __device__
        void merge(const BoundingBox& bBox)	{
            if (bBox.bEmpty) return;
            
            if (bEmpty || bBox.bmin.x < bmin.x) bmin.x = bBox.bmin.x;
            if (bEmpty || bBox.bmax.x > bmax.x) bmax.x = bBox.bmax.x;

            if (bEmpty || bBox.bmin.y < bmin.y) bmin.y = bBox.bmin.y;
            if (bEmpty || bBox.bmax.y > bmax.y) bmax.y = bBox.bmax.y;
            
            if (bEmpty || bBox.bmin.z < bmin.z) bmin.z = bBox.bmin.z;
            if (bEmpty || bBox.bmax.z > bmax.z) bmax.z = bBox.bmax.z;
            
            dirty = true;
            bEmpty = false;
        }
};
