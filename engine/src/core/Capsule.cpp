#include "../../include/core/Capsule.h"

using namespace PhysicsEngine;

Capsule::Capsule() : mCentre(glm::vec3(0.0f, 0.0f, 0.0f)), mRadius(1.0f), mHeight(1.0f)
{
}

Capsule::Capsule(glm::vec3 centre, float radius, float height) : mCentre(centre), mRadius(radius), mHeight(height)
{
}

Capsule::~Capsule()
{
}