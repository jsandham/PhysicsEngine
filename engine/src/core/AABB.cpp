#include <algorithm>
#include <iostream>
#include <limits>

#include "../../include/core/AABB.h"

using namespace PhysicsEngine;

AABB::AABB() : mCentre(glm::vec3(0.0f, 0.0f, 0.0f)), mSize(glm::vec3(1.0f, 1.0f, 1.0f))
{
}

AABB::AABB(const glm::vec3 &centre, const glm::vec3 &size) : mCentre(centre), mSize(size)
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

float AABB::getHalfSurfaceArea() const
{
    return mSize.x * mSize.y + mSize.y * mSize.z + mSize.z * mSize.x;
}

float AABB::getSurfaceArea() const
{
    return 2.0f * getHalfSurfaceArea();
}