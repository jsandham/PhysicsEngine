#include "../../include/core/Ray.h"

using namespace PhysicsEngine;

Ray::Ray()
{

}

Ray::Ray(glm::vec3 origin, glm::vec3 direction)
{
	mOrigin = origin;
	mDirection = direction;
}

Ray::~Ray()
{

}

glm::vec3 Ray::getPoint(float distance) const
{
	return mOrigin + distance *mDirection;
}