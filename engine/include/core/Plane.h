#ifndef PLANE_H__
#define PLANE_H__

#include "glm.h"

namespace PhysicsEngine
{
// plane defined by n.x*x + n.y*y + n.z*z + d = 0, where d = -dot(n, x0)
class Plane
{
  public:
    glm::vec3 mNormal;
    glm::vec3 mX0;

  public:
    Plane();
    Plane(glm::vec3 normal, glm::vec3 x0);

    float getD() const;
    float signedDistance(const glm::vec3 &point) const;
};
} // namespace PhysicsEngine

#endif