#include "../../include/core/Sphere.h"

using namespace PhysicsEngine;

Sphere::Sphere()
{
	this->centre = glm::vec3(0.0f, 0.0f, 0.0f);
	this->radius = 1.0f;
}

Sphere::Sphere(glm::vec3 centre, float radius)
{
	this->centre = centre;
	this->radius = radius;
}

Sphere::~Sphere()
{

}