#ifndef CIRCLE_H__
#define CIRCLE_H__

#include "glm.h"

namespace PhysicsEngine
{
class Circle
{
  public:
    glm::vec3 mCentre;
    glm::vec3 mNormal;
    float mRadius;

  public:
    Circle();
    Circle(glm::vec3 centre, glm::vec3 normal, float radius);

    float getArea() const;
    float getCircumference() const;
};
} // namespace PhysicsEngine

#endif