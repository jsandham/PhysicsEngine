#include "../../include/core/Line.h"

using namespace PhysicsEngine;

Line::Line() : mStart(glm::vec3(0.0f, 0.0f, 0.0f)), mEnd(glm::vec3(1.0f, 0.0f, 0.0f))
{
}

Line::Line(glm::vec3 start, glm::vec3 end) : mStart(start), mEnd(end)
{
}

float Line::getLength() const
{
    return glm::distance(mStart, mEnd);
}