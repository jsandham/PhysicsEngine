#ifndef TRIANGLE_H__
#define TRIANGLE_H__

#include "glm.h"
#include "AABB.h"

namespace PhysicsEngine
{
class Triangle
{
  public:
    glm::vec3 mV0;
    glm::vec3 mV1;
    glm::vec3 mV2;

  public:
    Triangle();
    Triangle(const glm::vec3 &v0, const glm::vec3 &v1, const glm::vec3 &v2);

    glm::vec3 getBarycentric(const glm::vec3 &p) const;
    glm::vec3 getCentroid() const;
    glm::vec3 getNormal() const;
    glm::vec3 getUnitNormal() const;
    AABB getAABBBounds() const;
};
} // namespace PhysicsEngine

#endif