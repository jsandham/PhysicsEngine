#ifndef CLOSEST_DISTANCE_H__
#define CLOSEST_DISTANCE_H__

#include "Circle.h"
#include "Ray.h"
#include "Sphere.h"

namespace PhysicsEngine
{
class ClosestDistance
{
  public:
    static float closestDistance(const Ray &ray1, const Ray &ray2, float &t1, float &t2);
    static float closestDistance(const Ray &ray, const Circle &circle, float &t, glm::vec3 &circlePoint);
    static float closestDistance(const Ray &ray, const Sphere &sphere, float &t, glm::vec3 &spherePoint);
};
} // namespace PhysicsEngine

#endif