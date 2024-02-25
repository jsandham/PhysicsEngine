#ifndef RECT_H__
#define RECT_H__

#include "glm.h"

namespace PhysicsEngine
{
class Rect
{
  public:
    float mX;
    float mY;
    float mWidth;
    float mHeight;

  public:
    Rect();
    Rect(float x, float y, float width, float height);
    Rect(const glm::vec2 &min, const glm::vec2 &max);

    bool contains(float x, float y) const;
    float getArea() const;
    glm::vec2 getCentre() const;
    glm::vec2 getMin() const;
    glm::vec2 getMax() const;
};
} // namespace PhysicsEngine

#endif