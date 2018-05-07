#include <iostream>
#include <limits>
#include <algorithm>

#include "../glm/glm.hpp"

#include "Geometry.h"

using namespace PhysicsEngine;

//see "3D Game Engine Design: A Practical Approach to Real-Time Computer Graphics" by David H Eberly
bool Geometry::intersect(Ray ray, Sphere sphere)
{
	ray.direction = glm::normalize(ray.direction);

	// quadratic of form 0 = t^2 + 2*a1*t + a0
	glm::vec3 q = ray.origin - sphere.centre;
	float a0 = glm::dot(q, q) - sphere.radius*sphere.radius;
	if (a0 <= 0){
		// ray origin stats inside the sphere
		return true;
	}

	float a1 = glm::dot(ray.direction, q);
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

	glm::vec3 min = bounds.centre - 0.5f*bounds.size;
	glm::vec3 max = bounds.centre + 0.5f*bounds.size;

	float tx0 = (min.x - ray.origin.x) / ray.direction.x;
	float tx1 = (max.x - ray.origin.x) / ray.direction.x;

	tmin = std::max(tmin, std::min(tx0, tx1));
	tmax = std::min(tmax, std::max(tx0, tx1));

	float ty0 = (min.y - ray.origin.y) / ray.direction.y;
	float ty1 = (max.y - ray.origin.y) / ray.direction.y;

	tmin = std::max(tmin, std::min(ty0, ty1));
	tmax = std::min(tmax, std::max(ty0, ty1));

	float tz0 = (min.z - ray.origin.z) / ray.direction.z;
	float tz1 = (max.z - ray.origin.z) / ray.direction.z;

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

	float radiusSqr = sphere.radius * sphere.radius;
	float distSqr = 0.0f;
	for (int i = 0; i < 3; i++){
		if (sphere.centre[i] < min[i]){
			distSqr += (sphere.centre[i] - min[i]) * (sphere.centre[i] - min[i]);
		}
		else if (sphere.centre.x > max.x){
			distSqr += (sphere.centre[i] - max[i]) * (sphere.centre[i] - max[i]);
		}
	}
	
	return distSqr <= radiusSqr;
}

bool Geometry::intersect(Sphere sphere1, Sphere sphere2)
{
	float distance = glm::length(sphere1.centre - sphere2.centre);

	return distance <= (sphere1.radius + sphere2.radius);
}

bool Geometry::intersect(Sphere sphere, Capsule capsule)
{
	return true;
}

bool Geometry::intersect(Bounds bounds1, Bounds bounds2)
{
	bool overlap_x = std::abs(bounds1.centre.x - bounds2.centre.x) <= 0.5f * (bounds1.size.x + bounds2.size.x);
	bool overlap_y = std::abs(bounds1.centre.y - bounds2.centre.y) <= 0.5f * (bounds1.size.y + bounds2.size.y);
	bool overlap_z = std::abs(bounds1.centre.z - bounds2.centre.z) <= 0.5f * (bounds1.size.z + bounds2.size.z);

	return overlap_x && overlap_y && overlap_z;
}

bool Geometry::intersect(Bounds bounds, Capsule capsule)
{
	return true;
}