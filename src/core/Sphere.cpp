#include "Sphere.h"

using namespace PhysicsEngine;

Sphere::Sphere()
{

}

Sphere::Sphere(glm::vec3 centre, float radius)
{
	this->centre = centre;
	this->radius = radius;
}

Sphere::~Sphere()
{

}