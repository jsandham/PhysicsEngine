#ifndef INTERSECT_H__
#define INTERSECT_H__

#include "AABB.h"
#include "Capsule.h"
#include "Frustum.h"
#include "Plane.h"
#include "Ray.h"
#include "Sphere.h"
#include "Triangle.h"

namespace PhysicsEngine
{
class Intersect
{
  private:
    static float EPSILON;

  public:
    static bool intersect(const Ray &ray, const Plane &plane, float &dist);

    static bool intersect(const Ray &ray, const Triangle &triangle);
    static bool intersect(const Ray &ray, const Plane &plane);
    static bool intersect(const Ray &ray, const Sphere &sphere);
    static bool intersect(const Ray &ray, const AABB &aabb);
    static bool intersect(const Ray &ray, const Capsule &capsule);
    static bool intersect(const Ray &ray, const Frustum &frustum);
    static bool intersect(const Sphere &sphere, const Sphere &sphere2);
    static bool intersect(const Sphere &sphere, const AABB &aabb);
    static bool intersect(const Sphere &sphere, const Capsule &capsule);
    static bool intersect(const Sphere &sphere, const Frustum &frustum);
    static bool intersect(const AABB &aabb, const AABB &aabb2);
    static bool intersect(const AABB &aabb, const Capsule &capsule);
    static bool intersect(const AABB &aabb, const Frustum &frustum);
    static bool intersect(const Frustum &frustum, const Frustum &frustum2);
};
} // namespace PhysicsEngine

#endif