#include "../../include/core/Sphere.h"

#include "glm/gtc/constants.hpp"

using namespace PhysicsEngine;

Sphere::Sphere() : mCentre(glm::vec3(0.0f, 0.0f, 0.0f)), mRadius(1.0f)
{
}

Sphere::Sphere(const glm::vec3 &centre, float radius) : mCentre(centre), mRadius(radius)
{
}

float Sphere::getVolume() const
{
    return (4.0f / 3.0f) * glm::pi<float>() * mRadius * mRadius * mRadius;
}

glm::vec3 Sphere::getNormal(const glm::vec3 &point) const
{
    return (point - mCentre) / mRadius;
}

glm::vec3 Sphere::getUnitNormal(const glm::vec3 &point) const
{
    return glm::normalize(getNormal(point));
}