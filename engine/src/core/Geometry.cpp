#include <iostream>
#include <limits>
#include <algorithm>

#include "../../include/glm/glm.hpp"

#include "../../include/core/Geometry.h"

using namespace PhysicsEngine;

//see "3D Game Engine Design: A Practical Approach to Real-Time Computer Graphics" by David H Eberly
bool Geometry::intersect(Ray ray, Sphere sphere)
{
	ray.mDirection = glm::normalize(ray.mDirection);

	// quadratic of form 0 = t^2 + 2*a1*t + a0
	glm::vec3 q = ray.mOrigin - sphere.mCentre;
	float a0 = glm::dot(q, q) - sphere.mRadius * sphere.mRadius;
	if (a0 <= 0){
		// ray origin stats inside the sphere
		return true;
	}

	float a1 = glm::dot(ray.mDirection, q);
	if (a1 >= 0){
		return false;
	}

	//if (a1 <= -maxDistance){
	//	return false;
	//}

	return (a1*a1 >= a0);
}

bool Geometry::intersect(Ray ray, Bounds bounds)
{
	float tmin = -1 * std::numeric_limits<float>::infinity();
	float tmax = std::numeric_limits<float>::infinity();

	glm::vec3 min = bounds.mCentre - 0.5f*bounds.mSize;
	glm::vec3 max = bounds.mCentre + 0.5f*bounds.mSize;

	float tx0 = (min.x - ray.mOrigin.x) / ray.mDirection.x;
	float tx1 = (max.x - ray.mOrigin.x) / ray.mDirection.x;

	tmin = std::max(tmin, std::min(tx0, tx1));
	tmax = std::min(tmax, std::max(tx0, tx1));

	float ty0 = (min.y - ray.mOrigin.y) / ray.mDirection.y;
	float ty1 = (max.y - ray.mOrigin.y) / ray.mDirection.y;

	tmin = std::max(tmin, std::min(ty0, ty1));
	tmax = std::min(tmax, std::max(ty0, ty1));

	float tz0 = (min.z - ray.mOrigin.z) / ray.mDirection.z;
	float tz1 = (max.z - ray.mOrigin.z) / ray.mDirection.z;

	tmin = std::max(tmin, std::min(tz0, tz1));
	tmax = std::min(tmax, std::max(tz0, tz1));

	return tmax >= tmin && tmax >= 0.0f;
}

bool Geometry::intersect(Ray ray, Capsule capsule)
{
	return true;
}

bool Geometry::intersect(Sphere sphere, Bounds bounds)
{
	glm::vec3 min = bounds.getMin();
	glm::vec3 max = bounds.getMax();

	float radiusSqr = sphere.mRadius * sphere.mRadius;
	float distSqr = 0.0f;
	for (int i = 0; i < 3; i++){
		if (sphere.mCentre[i] < min[i]){
			distSqr += (sphere.mCentre[i] - min[i]) * (sphere.mCentre[i] - min[i]);
		}
		else if (sphere.mCentre[i] > max[i]){
			distSqr += (sphere.mCentre[i] - max[i]) * (sphere.mCentre[i] - max[i]);
		}
	}
	
	return distSqr <= radiusSqr;


	// same algorithm probably faster?
	// glm::vec3 min = bounds.getMin();
	// glm::vec3 max = bounds.getMax();

	// float ex = std::max(min.x - center.x, 0.0f) + std::max(center.x - max.x, 0.0f);
	// float ey = std::max(min.y - center.y, 0.0f) + std::max(center.y - max.y, 0.0f);
	// float ez = std::max(min.z - center.z, 0.0f) + std::max(center.z - max.z, 0.0f);

	// return (ex < sphere.radius) && (ey < sphere.radius) && (ez < sphere.radius) && (ex * ex + ey * ey + ez * ez < sphere.radius * sphere.radius);
}

bool Geometry::intersect(Sphere sphere1, Sphere sphere2)
{
	float distance = glm::length(sphere1.mCentre - sphere2.mCentre);

	return distance <= (sphere1.mRadius + sphere2.mRadius);
}

bool Geometry::intersect(Sphere sphere, Capsule capsule)
{
	return true;
}

bool Geometry::intersect(Bounds bounds1, Bounds bounds2)
{
	bool overlap_x = std::abs(bounds1.mCentre.x - bounds2.mCentre.x) <= 0.5f * (bounds1.mSize.x + bounds2.mSize.x);
	bool overlap_y = std::abs(bounds1.mCentre.y - bounds2.mCentre.y) <= 0.5f * (bounds1.mSize.y + bounds2.mSize.y);
	bool overlap_z = std::abs(bounds1.mCentre.z - bounds2.mCentre.z) <= 0.5f * (bounds1.mSize.z + bounds2.mSize.z);

	return overlap_x && overlap_y && overlap_z;

	// same algorithm slightly re-written
	// glm::vec3 d = bounds1.center - bounds2.center;

	// float ex = Math.Abs(d.x) - (bounds1.halfsize.x + bounds2.halfsize.x);
	// float ey = Math.Abs(d.y) - (bounds1.halfsize.y + bounds2.halfsize.y);
	// float ez = Math.Abs(d.z) - (bounds1.halfsize.z + bounds2.halfsize.z);

	// return (ex < 0) && (ey < 0) && (ez < 0);
}

bool Geometry::intersect(Bounds bounds, Capsule capsule)
{
	return true;
}