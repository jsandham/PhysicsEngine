#include <limits>
#include <algorithm>

#include "../../include/core/Intersect.h"

#include "../../include/glm/glm.hpp"

using namespace PhysicsEngine;

float Intersect::EPSILON = 0.0001f;

// Moller-Trumbore algorithm
//https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection
bool Intersect::intersect(Ray ray, Triangle triangle)
{
	glm::vec3 v0v1 = triangle.v1 - triangle.v0;
	glm::vec3 v0v2 = triangle.v2 - triangle.v0;
	glm::vec3 pvec = glm::cross(ray.mDirection, v0v2);
	float det = glm::dot(v0v1, pvec);
#ifdef CULLING 
	// if the determinant is negative the triangle is backfacing
	// if the determinant is close to 0, the ray misses the triangle
	if (det < EPSILON) return false;
#else 
	// ray and triangle are parallel if det is close to 0
	if (fabs(det) < EPSILON) return false;
#endif 
	float invDet = 1 / det;

	glm::vec3 tvec = ray.mOrigin - triangle.v0;
	float u = glm::dot(tvec, pvec) * invDet;
	if (u < 0 || u > 1) return false;

	glm::vec3 qvec = glm::cross(tvec, v0v1);
	float v = glm::dot(ray.mDirection, qvec) * invDet;
	if (v < 0 || u + v > 1) return false;

	//float t = glm::dot(v0v2, qvec) * invDet;

	return true;
}

bool Intersect::intersect(Ray ray, Plane plane)
{
	float denom = glm::dot(plane.mNormal, ray.mDirection);
	if (abs(denom) > EPSILON)
	{
		if (glm::dot(plane.mX0 - ray.mOrigin, plane.mNormal) / denom >= 0) {
			return true;
		}
	}

	return false;
}

//see "3D Game Engine Design: A Practical Approach to Real-Time Computer Graphics" by David H Eberly
bool Intersect::intersect(Ray ray, Sphere sphere)
{
	ray.mDirection = glm::normalize(ray.mDirection);

	// quadratic of form 0 = t^2 + 2*a1*t + a0
	glm::vec3 q = ray.mOrigin - sphere.mCentre;
	float a0 = glm::dot(q, q) - sphere.mRadius * sphere.mRadius;
	if (a0 <= 0) {
		// ray origin stats inside the sphere
		return true;
	}

	float a1 = glm::dot(ray.mDirection, q);
	if (a1 >= 0) {
		return false;
	}

	//if (a1 <= -maxDistance){
	//	return false;
	//}

	return (a1 * a1 >= a0);
}

bool Intersect::intersect(Ray ray, AABB aabb)
{
	glm::vec3 min = aabb.mCentre - 0.5f * aabb.mSize;
	glm::vec3 max = aabb.mCentre + 0.5f * aabb.mSize;

	glm::vec3 invDirection;
	invDirection.x = 1.0f / ray.mDirection.x;
	invDirection.y = 1.0f / ray.mDirection.y;
	invDirection.z = 1.0f / ray.mDirection.z;

	float tx0 = (min.x - ray.mOrigin.x) * invDirection.x;
	float tx1 = (max.x - ray.mOrigin.x) * invDirection.x;
	float ty0 = (min.y - ray.mOrigin.y) * invDirection.y;
	float ty1 = (max.y - ray.mOrigin.y) * invDirection.y;
	float tz0 = (min.z - ray.mOrigin.z) * invDirection.z;
	float tz1 = (max.z - ray.mOrigin.z) * invDirection.z;

	float tmin = std::max(std::max(std::min(tx0, tx1), std::min(ty0, ty1)), std::min(tz0, tz1));
	float tmax = std::min(std::min(std::max(tx0, tx1), std::max(ty0, ty1)), std::max(tz0, tz1));

	return tmax >= tmin && tmax >= 0.0f;
}

bool Intersect::intersect(Ray ray, Capsule capsule)
{
	return false;
}

bool Intersect::intersect(Ray ray, Frustum frustum)
{
	if (frustum.containsPoint(ray.mOrigin)) {
		return true;
	}


	return false;
}

bool Intersect::intersect(Sphere sphere, Sphere sphere2)
{
	float distance = glm::length(sphere.mCentre - sphere2.mCentre);

	return distance <= (sphere.mRadius + sphere2.mRadius);
}

bool Intersect::intersect(Sphere sphere, AABB aabb)
{
	glm::vec3 min = aabb.getMin();
	glm::vec3 max = aabb.getMax();

	float ex = std::max(min.x - sphere.mCentre.x, 0.0f) + std::max(sphere.mCentre.x - max.x, 0.0f);
	float ey = std::max(min.y - sphere.mCentre.y, 0.0f) + std::max(sphere.mCentre.y - max.y, 0.0f);
	float ez = std::max(min.z - sphere.mCentre.z, 0.0f) + std::max(sphere.mCentre.z - max.z, 0.0f);

	return (ex < sphere.mRadius) && (ey < sphere.mRadius) && (ez < sphere.mRadius) && (ex * ex + ey * ey + ez * ez < sphere.mRadius* sphere.mRadius);
}

bool Intersect::intersect(Sphere sphere, Capsule capsule)
{
	return true;
}

bool Intersect::intersect(Sphere sphere, Frustum frustum)
{
	// sphere lies outside frustum
	if (frustum.mPlanes[0].signedDistance(sphere.mCentre) < -sphere.mRadius) { return false; }
	if (frustum.mPlanes[1].signedDistance(sphere.mCentre) < -sphere.mRadius) { return false; }
	if (frustum.mPlanes[2].signedDistance(sphere.mCentre) < -sphere.mRadius) { return false; }
	if (frustum.mPlanes[3].signedDistance(sphere.mCentre) < -sphere.mRadius) { return false; }
	if (frustum.mPlanes[4].signedDistance(sphere.mCentre) < -sphere.mRadius) { return false; }
	if (frustum.mPlanes[5].signedDistance(sphere.mCentre) < -sphere.mRadius) { return false; }

	// sphere touches or intersect frustum
	return true;
}

bool Intersect::intersect(AABB aabb, AABB aabb2)
{
	bool overlap_x = std::abs(aabb.mCentre.x - aabb2.mCentre.x) <= 0.5f * (aabb.mSize.x + aabb2.mSize.x);
	bool overlap_y = std::abs(aabb.mCentre.y - aabb2.mCentre.y) <= 0.5f * (aabb.mSize.y + aabb2.mSize.y);
	bool overlap_z = std::abs(aabb.mCentre.z - aabb2.mCentre.z) <= 0.5f * (aabb.mSize.z + aabb2.mSize.z);

	return overlap_x && overlap_y && overlap_z;

	// same algorithm slightly re-written
	// glm::vec3 d = aabb1.center - aabb2.center;

	// float ex = Math.Abs(d.x) - (aabb1.halfsize.x + aabb2.halfsize.x);
	// float ey = Math.Abs(d.y) - (aabb1.halfsize.y + aabb2.halfsize.y);
	// float ez = Math.Abs(d.z) - (aabb1.halfsize.z + aabb2.halfsize.z);

	// return (ex < 0) && (ey < 0) && (ez < 0);
}

bool Intersect::intersect(AABB aabb, Capsule capsule)
{
	return true;
}

bool Intersect::intersect(AABB aabb, Frustum frustum)
{
	for (int i = 0; i < 6; i++) {

		// maximum extent in direction of plane normal 
		float r = fabsf(aabb.mSize.x * frustum.mPlanes[i].mNormal.x)
			+ fabsf(aabb.mSize.y * frustum.mPlanes[i].mNormal.y)
			+ fabsf(aabb.mSize.z * frustum.mPlanes[i].mNormal.z);

		// signed distance between box center and plane
		float d = frustum.mPlanes[i].signedDistance(aabb.mCentre);

		// return signed distance
		float side = d - r;
		if (fabsf(d) < r) {
			side = 0.0f;
		}
		else if (d < 0.0f) {
			side = d + r;
		}

		if (side < 0) {
			return false;
		}
	}

	return true;
}