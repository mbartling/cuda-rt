#pragma once
#include <cuda_runtime.h>
#include <math.h>

class Vec3d{
    public:
        double x;
        double y;
        double z;

        // Constructors
        __host__ __device__
        Vec3d(double x, double y, double z): x(x), y(y), z(z){}
        
        __host__ __device__
        Vec3d(double x, double y): x(x), y(y) {}

        __host__ __device__
        explicit Vec3d(double a): x(a), y(a), z(a) {}
        
        __host__ __device__
        explicit Vec3d(double* a) : x(a[0]), y(a[1]), z(a[2]) {}

        __host__ __device__
        Vec3d(void): x(0.0), y(0.0), z(0.0) {}

        __host__ __device__
        Vec3d& operator = (double a){ x = a; y = a; z = a; return *this; }

        __host__ __device__
        Vec3d& operator /= (double a){ x /= a; y /= a; z /= a; return *this; }
        __host__ __device__
        Vec3d& operator -= (double a){ x -= a; y -= a; z -= a; return *this; }
        
        __host__ __device__
        Vec3d operator - () const { return Vec3d(-x, -y, -z); }
        
        __host__ __device__
        Vec3d& operator *= (double a){ x *= a; y *= a; z *= a; return *this; }
        
        __host__ __device__
        Vec3d& operator %= (const Vec3d& a){ x *= a.x; y *= a.y; z *= a.z; return *this; }
        
        __host__ __device__
        Vec3d& operator -= (const Vec3d& a){ x -= a.x; y -= a.y; z -= a.z; return *this; }
        __host__ __device__
        Vec3d& operator += (const Vec3d& a){ x += a.x; y += a.y; z += a.z; return *this; }

        __host__ __device__
        double operator [] (int i) const { if(i == 0){ return x; } if(i == 1){ return y; }if(i == 2){ return z; } else{return 1.0e307; }}


};

__host__ __device__ __inline__
double operator * (const Vec3d& a, const Vec3d& b){
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

__host__ __device__ __inline__

Vec3d operator + (const Vec3d& a, const Vec3d& b){
    return Vec3d(a.x + b.x , a.y + b.y , a.z + b.z);
}

__host__ __device__ __inline__
Vec3d operator - (const Vec3d& a, const Vec3d& b){
    return Vec3d(a.x - b.x , a.y - b.y , a.z - b.z);
}

__host__ __device__ __inline__
Vec3d operator * (const Vec3d& a, double b){
    return Vec3d(a.x*b, a.y*b, a.z*b);
}

__host__ __device__ __inline__
Vec3d operator - (const Vec3d& a, double b){
    return Vec3d(a.x - b, a.y -b, a.z -b);
}

__host__ __device__ __inline__
Vec3d operator + (const Vec3d& a, double b){
    return Vec3d(a.x + b, a.y+b, a.z+b);
}

__host__ __device__ __inline__
Vec3d operator * (double b, const Vec3d& a){
    return Vec3d(a.x*b, a.y*b, a.z*b);
}

__host__ __device__ __inline__
Vec3d operator - (double b, const Vec3d& a){
    return Vec3d(a.x - b, a.y -b, a.z -b);
}

__host__ __device__ __inline__
Vec3d operator + (double b, const Vec3d& a){
    return Vec3d(a.x + b, a.y+b, a.z+b);
}
    __host__ __device__ __inline__
Vec3d operator / (const Vec3d& a, double b){
    return Vec3d(a.x/b, a.y/b, a.z/b);
}

__host__ __device__ __inline__
Vec3d operator % (const Vec3d& a, const Vec3d& b){
    return Vec3d(a.x * b.x , a.y * b.y , a.z * b.z);
}
//Cross Product
__host__ __device__ __inline__
Vec3d operator ^ (const Vec3d& a, const Vec3d& b){
    return Vec3d(a.y*b.z - a.z*b.y,
                 a.z*b.x - a.x*b.z,
                 a.x*b.y - a.y*b.x);
}

__host__ __device__ __inline__
Vec3d& normalize(Vec3d& a){
    double sqinv = 1.0/sqrtf(a.x*a.x + a.y*a.y + a.z*a.z); //rnorm3df(a.x, a.y, a.z); TODO where is this defined
    a *= sqinv;
    return a;
}
__host__ __device__ __inline__
void clamp(double& a){
    if (a < 0) a = 0.0;
    if (a > 1) a = 1.0;
}
__host__ __device__ __inline__
Vec3d& clamp(Vec3d& a){
    clamp(a.x);
    clamp(a.y);
    clamp(a.z);
    return a;
}

__host__ __device__ __inline__
double norm(const Vec3d& a){
    //return norm3df(a.x, a.y, a.z);
    return sqrtf(a.x*a.x + a.y*a.y + a.z*a.z);
}

__host__ __device__ __inline__
double rnorm(const Vec3d& a){
    double sqinv = 1.0/sqrtf(a.x*a.x + a.y*a.y + a.z*a.z); //rnorm3df(a.x, a.y, a.z); TODO where is this defined
    return sqinv;
}

__host__ __device__ __inline__
Vec3d recip(const Vec3d& a){
    return Vec3d(1.0/a.x, 1.0/a.y, 1.0/a.z);
}

__host__ __device__ __inline__
bool isZero(const Vec3d& a){
    return (a.x == 0) && (a.y == 0) && (a.z == 0);
}
__host__ __device__ __inline__
Vec3d maximum(const Vec3d& a, const Vec3d& b){
    return Vec3d(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

__host__ __device__ __inline__
Vec3d minimum(const Vec3d& a, const Vec3d& b){
    return Vec3d(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

class Vec4d{
    public:
        double x;
        double y;
        double z;
        double w;
        
        // Constructors
        __host__ __device__
        Vec4d(double x, double y, double z, double w): x(x), y(y), z(z), w(w){}

        __host__ __device__
        Vec4d(double x, double y, double z): x(x), y(y), z(z){}
        
        __host__ __device__
        Vec4d(double x, double y): x(x), y(y) {}

        __host__ __device__
        Vec4d(double a): x(a), y(a), z(a), w(a) {}
        
        __host__ __device__
        Vec4d(): x(0.0), y(0.0), z(0.0), w(0.0) {}

        __host__ __device__
        Vec4d& operator = (double a){ x = a; y = a; z = a; w = a; return *this; }
        
        __host__ __device__
        Vec4d operator - (){ return Vec4d(-x, -y, -z, -w); }
        
        __host__ __device__
        Vec4d& operator *= (double a){ x *= a; y *= a; z *= a; w *= a; return *this; }
        
        __host__ __device__
        Vec4d& operator %= (const Vec4d& a){ x *= a.x; y *= a.y; z *= a.z; w *= a.w; return *this; }

};

__host__ __device__ __inline__
double operator * (const Vec4d& a, const Vec4d b){
    return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
}

__host__ __device__ __inline__

Vec4d operator + (const Vec4d& a, const Vec4d b){
    return Vec4d(a.x + b.x , a.y + b.y , a.z + b.z, a.w + b.w);
}

__host__ __device__ __inline__
Vec4d operator - (const Vec4d& a, const Vec4d b){
    return Vec4d(a.x - b.x , a.y - b.y , a.z - b.z, a.w - b.w);
}

__host__ __device__ __inline__
Vec4d operator % (const Vec4d& a, const Vec4d b){
    return Vec4d(a.x * b.x , a.y * b.y , a.z * b.z, a.w*b.w);
}

__host__ __device__ __inline__
void normalize(Vec4d& a){
    double sqinv = 1.0f/sqrtf(a.x*a.x + a.y*a.y + a.z*a.z + a.w*a.w);
    a *= sqinv;
}
__host__ __device__ __inline__
double norm(const Vec4d& a){
    return sqrtf(a.x*a.x + a.y*a.y + a.z*a.z + a.w*a.w);
}

// x,y,z are rows
class Mat3d{
    public:
        Vec3d x;
        Vec3d y;
        Vec3d z;

        __host__ __device__
        Mat3d(double xx, double xy, double xz,
              double yx, double yy, double yz,
              double zx, double zy, double zz): x(xx,yy,zz), y(yx, yy, yz), z(zx, zy, zz) {}
        
        __host__ __device__
        Mat3d(void) : x(0.0), y(0.0), z(0.0) {}
        
        __host__ __device__
        Mat3d(Vec3d x, Vec3d y, Vec3d z) : x(x), y(y), z(z) {}

};

__host__ __device__ __inline__
Vec3d multAT_x(const Mat3d& A, const Vec3d& f){
    return Vec3d(A.x.x * f.x + A.y.x * f.y + A.z.x * f.z, 
                 A.x.y * f.x + A.y.y*f.y + A.z.y*f.z, 
                 A.x.z * f.x + A.y.z * f.y + A.z.z * f.z);
}

__host__ __device__ __inline__
Vec3d multxT_A(const Vec3d& f, const Mat3d& A){
    return multAT_x(A, f); // Same value except this one should be treated as transposed
}
__host__ __device__ __inline__
Vec3d operator * (const Mat3d& a, const Vec3d& f){
    return Vec3d(a.x*f, a.y*f, a.z*f);
}
class Mat4d{
    public:
        Vec4d x; //Row 1
        Vec4d y; //Row 2
        Vec4d z;
        Vec4d w;

        __host__ __device__
        Mat4d(double xx, double xy, double xz, double xw,
              double yx, double yy, double yz, double yw,
              double zx, double zy, double zz, double zw): x(xx,yy,zz,zw), y(yx, yy, yz,yw), z(zx, zy, zz, zw) {}
        
        
        __host__ __device__
        Mat4d(const Vec4d& x, const Vec4d& y, const Vec4d& z, const Vec4d& w) : x(x), y(y), z(z), w(w) {}
        
        __host__ __device__
        Mat4d(): x(), y(), z(), w() {}
        
        __host__ __device__
        Mat4d& operator *= (double d){
            x *= d;
            y *= d;
            z *= d;
            w *= d;
            return *this;
        }

};

__host__ __device__ __inline__
double operator * (const Vec4d& a, const Vec3d& b){
    return a.x*b.x + a.y*b.y + a.z*b.z + a.w;
}
__host__ __device__ __inline__
Vec3d operator * (const Mat4d& a, const Vec3d& b){
    return Vec3d(a.x*b, a.y*b, a.z*b);
}

//__host__ __device__
//Mat4d getInverse(const Mat4d& a)
//{
//    double s0 = a.x.x * a.y.y - a.y.x * a.x.y;
//    double s1 = a.x.x * a.y.z - a.y.x * a.x.z;
//    double s2 = a.x.x * a.y.w - a.y.x * a.x.w;
//    double s3 = a.x.y * a.y.z - a.y.y * a.x.z;
//    double s4 = a.x.y * a.y.w - a.y.y * a.x.w;
//    double s5 = a.x.z * a.y.w - a.y.z * a.x.w;
//
//    double c5 = a.z.z * a.w.w - a.w.z * a.z.w;
//    double c4 = a.z.y * a.w.w - a.w.y * a.z.w;
//    double c3 = a.z.y * a.w.z - a.w.y * a.z.z;
//    double c2 = a.z.x * a.w.w - a.w.x * a.z.w;
//    double c1 = a.z.x * a.w.z - a.w.x * a.z.z;
//    double c0 = a.z.x * a.w.y - a.w.x * a.z.y;
//
//    // Should check for 0 determinant
//    double invdet = 1.0 / (s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0);
//
//    Mat4d b;
//
//    b.x = Vec4d(( a.y.y * c5 - a.y.z * c4 + a.y.w * c3) 
//          ,(-a.x.y * c5 + a.x.z * c4 - a.x.w * c3) 
//          ,( a.w.y * s5 - a.w.z * s4 + a.w.w * s3) 
//          ,(-a.z.y * s5 + a.z.z * s4 - a.z.w * s3)) ;
//
//    b.y = Vec4d((-a.y.x * c5 + a.y.z * c2 - a.y.w * c1) 
//          ,( a.x.x * c5 - a.x.z * c2 + a.x.w * c1) 
//          ,(-a.w.x * s5 + a.w.z * s2 - a.w.w * s1) 
//          ,( a.z.x * s5 - a.z.z * s2 + a.z.w * s1)) ;
//
//    b.z = Vec4d(( a.y.x * c4 - a.y.y * c2 + a.y.w * c0) 
//          ,(-a.x.x * c4 + a.x.y * c2 - a.x.w * c0) 
//          ,( a.w.x * s4 - a.w.y * s2 + a.w.w * s0) 
//          ,(-a.z.x * s4 + a.z.y * s2 - a.z.w * s0)) ;
//
//    b.w = Vec4d((-a.y.x * c3 + a.y.y * c1 - a.y.z * c0) 
//          ,( a.x.x * c3 - a.x.y * c1 + a.x.z * c0) 
//          ,(-a.w.x * s3 + a.w.y * s1 - a.w.z * s0) 
//          ,( a.z.x * s3 - a.z.y * s1 + a.z.z * s0)) ;
//    
//    b*= invdet;
//    return b;
//}
