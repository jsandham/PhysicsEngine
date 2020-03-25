#include "../../include/core/Sphere.h"

using namespace PhysicsEngine;

Sphere::Sphere()
{
	mCentre = glm::vec3(0.0f, 0.0f, 0.0f);
	mRadius = 1.0f;
}

Sphere::Sphere(glm::vec3 centre, float radius)
{
	mCentre = centre;
	mRadius = radius;
}

Sphere::~Sphere()
{

}