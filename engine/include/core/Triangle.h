#ifndef TRIANGLE_H__
#define TRIANGLE_H__

#include "../glm/glm.hpp"

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
    Triangle(glm::vec3 v0, glm::vec3 v1, glm::vec3 v2);
    ~Triangle();
};
} // namespace PhysicsEngine

#endif