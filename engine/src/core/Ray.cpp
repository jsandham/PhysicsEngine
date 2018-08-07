#include "../../include/core/Ray.h"

using namespace PhysicsEngine;

Ray::Ray()
{

}

Ray::Ray(glm::vec3 origin, glm::vec3 direction)
{
	this->origin = origin;
	this->direction = direction;
}

Ray::~Ray()
{

}

glm::vec3 Ray::getPoint(float distance)
{
	return origin + distance * direction;
}