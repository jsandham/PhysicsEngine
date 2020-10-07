#ifndef __CLOSEST_DISTANCE_H__
#define __CLOSEST_DISTANCE_H__

#include "Circle.h"
#include "Ray.h"

namespace PhysicsEngine
{
class ClosestDistance
{
  public:
    static float closestDistance(Ray ray1, Ray ray2);
    static float closestDistance(Ray ray, Circle circle);
};
} // namespace PhysicsEngine

#endif