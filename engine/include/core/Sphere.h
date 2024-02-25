#ifndef SPHERE_H__
#define SPHERE_H__

#include "glm.h"

namespace PhysicsEngine
{
class Sphere
{
  public:
    glm::vec3 mCentre;
    float mRadius;

  public:
    Sphere();
    Sphere(const glm::vec3 &centre, float radius);

    float getVolume() const;
    glm::vec3 getNormal(const glm::vec3 &point) const;
    glm::vec3 getUnitNormal(const glm::vec3 &point) const;
};
} // namespace PhysicsEngine

#endif