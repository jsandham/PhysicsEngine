#include <iostream>
#include <limits>
#include <algorithm>

#include "../../include/core/AABB.h"

using namespace PhysicsEngine;

AABB::AABB() : mCentre(glm::vec3(0.0f, 0.0f, 0.0f)), mSize(glm::vec3(1.0f, 1.0f, 1.0f))
{

}

AABB::AABB(glm::vec3 centre, glm::vec3 size) : mCentre(centre), mSize(size)
{
	
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