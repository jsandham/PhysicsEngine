#include "../../include/core/Ray.h"

using namespace PhysicsEngine;

Ray::Ray() : mOrigin(glm::vec3(0.0f, 0.0f, 0.0f)), mDirection(glm::vec3(1.0f, 0.0f, 0.0f))
{
}

Ray::Ray(glm::vec3 origin, glm::vec3 direction) : mOrigin(origin), mDirection(direction)
{
}

Ray::~Ray()
{
}

glm::vec3 Ray::getPoint(float t) const
{
    return mOrigin + t * mDirection;
}