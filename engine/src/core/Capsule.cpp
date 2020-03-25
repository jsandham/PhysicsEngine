#include "../../include/core/Capsule.h"

using namespace PhysicsEngine;

Capsule::Capsule()
{

}

Capsule::Capsule(glm::vec3 centre, float radius, float height)
{
	mCentre = centre;
	mRadius = radius;
	mHeight = height;
}

Capsule::~Capsule()
{

}