#ifndef AABB_H__
#define AABB_H__

#include "glm.h"

namespace PhysicsEngine
{
class AABB
{
  public:
    glm::vec3 mCentre;
    glm::vec3 mSize;

  public:
    AABB();
    AABB(const glm::vec3 &centre, const glm::vec3 &size);

    glm::vec3 getExtents() const;
    glm::vec3 getMin() const;
    glm::vec3 getMax() const;
    float getHalfSurfaceArea() const;
    float getSurfaceArea() const;
};
} // namespace PhysicsEngine

#endif