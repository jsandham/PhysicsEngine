#include "../../include/core/Sphere.h"

#include "../../include/glm/gtc/constants.hpp"

using namespace PhysicsEngine;

Sphere::Sphere() : mCentre(glm::vec3(0.0f, 0.0f, 0.0f)), mRadius(1.0f)
{
}

Sphere::Sphere(const glm::vec3 &centre, float radius) : mCentre(centre), mRadius(radius)
{
}

Sphere::~Sphere()
{
}

float Sphere::getVolume() const
{
    return (4.0f / 3.0f) * glm::pi<float>() * mRadius * mRadius * mRadius;
}