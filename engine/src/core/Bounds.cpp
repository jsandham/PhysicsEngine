#include <iostream>
#include <limits>
#include <algorithm>

#include "../../include/core/Bounds.h"
#include "../../include/core/Geometry.h"

using namespace PhysicsEngine;

Bounds::Bounds()
{

}

Bounds::Bounds(glm::vec3 centre, glm::vec3 size)
{
	this->centre = centre;
	this->size = size;
}

Bounds::~Bounds()
{

}

glm::vec3 Bounds::getExtents() const
{
	return 0.5f * size;
}

glm::vec3 Bounds::getMin() const
{
	return centre - 0.5f * size;
}

glm::vec3 Bounds::getMax() const
{
	return centre + 0.5f * size;
}