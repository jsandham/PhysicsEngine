#include "../../include/core/Plane.h"

using namespace PhysicsEngine;

Plane::Plane() : mNormal(glm::vec3(1, 0, 0)), mX0(glm::vec3(0, 0, 0))
{
}

Plane::Plane(glm::vec3 normal, glm::vec3 x0) : mNormal(normal), mX0(x0)
{
}

float Plane::getD() const
{
    return -glm::dot(mNormal, mX0);
}

float Plane::signedDistance(const glm::vec3 &point) const
{
    float d = -glm::dot(mNormal, mX0); // could cache this

    //return (glm::dot(mNormal, point) + d) / sqrt(glm::dot(mNormal, mNormal));
    return glm::dot(mNormal, point) + d;
}