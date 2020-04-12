#include <iostream>
#include <limits>
#include <algorithm>

#include "../../include/core/AABB.h"
#include "../../include/core/Geometry.h"

using namespace PhysicsEngine;

AABB::AABB()
{

}

AABB::AABB(glm::vec3 centre, glm::vec3 size)
{
	mCentre = centre;
	mSize = size;
}

AABB::~AABB()
{

}

glm::vec3 AABB::getExtents() const
{
	return 0.5f * mSize;
}

glm::vec3 AABB::getMin() const
{
	return mCentre - 0.5f * mSize;
}

glm::vec3 AABB::getMax() const
{
	return mCentre + 0.5f * mSize;
}