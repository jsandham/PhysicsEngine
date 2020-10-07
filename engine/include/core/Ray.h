#ifndef __RAY_H__
#define __RAY_H__

#include "../glm/glm.hpp"

namespace PhysicsEngine
{
class Ray
{
  public:
    glm::vec3 mOrigin;
    glm::vec3 mDirection;

  public:
    Ray();
    Ray(glm::vec3 origin, glm::vec3 direction);
    ~Ray();

    glm::vec3 getPoint(float t) const;
};
} // namespace PhysicsEngine

#endif