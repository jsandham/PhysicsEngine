#ifndef LINE_H__
#define LINE_H__

#include "../glm/glm.hpp"

namespace PhysicsEngine
{
class Line
{
  public:
    glm::vec3 mStart;
    glm::vec3 mEnd;

  public:
    Line();
    Line(glm::vec3 start, glm::vec3 end);
    ~Line();

    float getLength() const;
};
} // namespace PhysicsEngine

#endif