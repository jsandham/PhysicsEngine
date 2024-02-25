#ifndef RAY_H__
#define RAY_H__

#include "glm.h"

namespace PhysicsEngine
{
class Ray
{
  public:
    glm::vec3 mOrigin;
    glm::vec3 mDirection;

  public:
    Ray();
    Ray(const glm::vec3 &origin, const glm::vec3 &direction);

    glm::vec3 getPoint(float t) const;
};
} // namespace PhysicsEngine

#endif