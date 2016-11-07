#include "ray.h"
#include <cmath>
#include <stdlib.h>

Vec3d CosWeightedRandomHemiDir2(Vec3d n){
  double Xi1 = (double)rand()/(double)RAND_MAX;
  double Xi2 = (double)rand()/(double)RAND_MAX;

  double theta = acos(sqrt(1.0-Xi1));
  double phi = 2.0*3.1415926535897932384626433832795 * Xi2;

  double xs = sin(theta) * cos(phi);
  double ys = cos(theta);
  double zs = sin(theta)*sin(phi);

//this doesn't compile comment it out for now
  Vec3d h = n;
  if(fabsf(h.x) <= fabsf(h.y) && fabsf(h.x) <= fabsf(h.z))
    h.x = 1.0;
  else if(fabsf(h.y) <= fabsf(h.x) && fabsf(h.y) <= fabsf(h.z))
    h.y = 1.0;
  else
    h.z = 1.0;

  Vec3d x = (h ^ n); 
  normalize(x);
  Vec3d z = (x ^ n); 
  normalize(z);
  Vec3d dir = xs*x + ys*n + zs*z;
  normalize(dir);
  return dir;
}
