#include "../../include/core/Capsule.h"

using namespace PhysicsEngine;

Capsule::Capsule()
{

}

Capsule::Capsule(glm::vec3 centre, float radius, float height)
{
	this->centre = centre;
	this->radius = radius;
	this->height = height;
}

Capsule::~Capsule()
{

}