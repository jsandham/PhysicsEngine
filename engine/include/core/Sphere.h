#ifndef __SPHERE_H__
#define __SPHERE_H__

#include "../glm/glm.hpp"

namespace PhysicsEngine
{
class Sphere
{
  public:
    float mRadius;
    glm::vec3 mCentre;

  public:
    Sphere();
    Sphere(const glm::vec3 &centre, float radius);
    ~Sphere();

    float getVolume() const;
};
} // namespace PhysicsEngine
#endif