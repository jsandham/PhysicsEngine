#ifndef CAPSULE_H__
#define CAPSULE_H__

#include "../glm/glm.hpp"

namespace PhysicsEngine
{
class Capsule
{
  public:
    glm::vec3 mCentre;
    float mRadius;
    float mHeight;

  public:
    Capsule();
    Capsule(glm::vec3 centre, float radius, float height);
    ~Capsule();
};
} // namespace PhysicsEngine

#endif