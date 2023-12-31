#include <algorithm>
#include <limits>

#include "../../include/core/Intersect.h"

using namespace PhysicsEngine;

float Intersect::EPSILON = 0.000001f;

bool Intersect::intersect(const Ray &ray, const Plane &plane, float &dist)
{
    float denom = glm::dot(plane.mNormal, ray.mDirection);
    if (abs(denom) > EPSILON)
    {
        dist = glm::dot(plane.mX0 - ray.mOrigin, plane.mNormal) / denom;
        return dist >= 0;
    }

    dist = 0.0f;
    return false;
}

// Moller-Trumbore algorithm
// https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection
bool Intersect::intersect(const Ray &ray, const Triangle &triangle)
{
    glm::vec3 v0v1 = triangle.mV1 - triangle.mV0;
    glm::vec3 v0v2 = triangle.mV2 - triangle.mV0;
    glm::vec3 pvec = glm::cross(ray.mDirection, v0v2);
    float det = glm::dot(v0v1, pvec);
#ifdef CULLING
    // if the determinant is negative the triangle is backfacing
    // if the determinant is close to 0, the ray misses the triangle
    if (det < EPSILON)
        return false;
#else
    // ray and triangle are parallel if det is close to 0
    if (fabs(det) < EPSILON)
        return false;
#endif
    float invDet = 1 / det;

    glm::vec3 tvec = ray.mOrigin - triangle.mV0;
    float u = glm::dot(tvec, pvec) * invDet;
    if (u < 0 || u > 1)
        return false;

    glm::vec3 qvec = glm::cross(tvec, v0v1);
    float v = glm::dot(ray.mDirection, qvec) * invDet;
    if (v < 0 || u + v > 1)
        return false;

    // float t = glm::dot(v0v2, qvec) * invDet;

    return true;
}

bool Intersect::intersect(const Ray &ray, const Plane &plane)
{
    float denom = glm::dot(plane.mNormal, ray.mDirection);
    if (abs(denom) > EPSILON)
    {
        if (glm::dot(plane.mX0 - ray.mOrigin, plane.mNormal) / denom >= 0)
        {
            return true;
        }
    }

    return false;
}

// see "3D Game Engine Design: A Practical Approach to Real-Time Computer Graphics" by David H Eberly
bool Intersect::intersect(const Ray &ray, const Sphere &sphere)
{
    // ray.mDirection = glm::normalize(ray.mDirection);

    // quadratic of form 0 = t^2 + 2*a1*t + a0
    glm::vec3 q = ray.mOrigin - sphere.mCentre;
    float a0 = glm::dot(q, q) - sphere.mRadius * sphere.mRadius;
    if (a0 <= 0)
    {
        // ray origin stats inside the sphere
        return true;
    }

    float a1 = glm::dot(ray.mDirection, q);
    if (a1 >= 0)
    {
        return false;
    }

    // if (a1 <= -maxDistance){
    //	return false;
    //}

    return (a1 * a1 >= a0);
}

bool Intersect::intersect(const Ray &ray, const AABB &aabb)
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

bool Intersect::intersect(const Ray &ray, const Frustum &frustum)
{
    if (frustum.containsPoint(ray.mOrigin))
    {
        return true;
    }

    return false;
}

bool Intersect::intersect(const Sphere &sphere, const Sphere &sphere2)
{
    float distance = glm::length(sphere.mCentre - sphere2.mCentre);

    return distance <= (sphere.mRadius + sphere2.mRadius);
}

bool Intersect::intersect(const Sphere &sphere, const AABB &aabb)
{
    glm::vec3 min = aabb.getMin();
    glm::vec3 max = aabb.getMax();

    float ex = std::max(min.x - sphere.mCentre.x, 0.0f) + std::max(sphere.mCentre.x - max.x, 0.0f);
    float ey = std::max(min.y - sphere.mCentre.y, 0.0f) + std::max(sphere.mCentre.y - max.y, 0.0f);
    float ez = std::max(min.z - sphere.mCentre.z, 0.0f) + std::max(sphere.mCentre.z - max.z, 0.0f);

    return (ex < sphere.mRadius) && (ey < sphere.mRadius) && (ez < sphere.mRadius) &&
           (ex * ex + ey * ey + ez * ez < sphere.mRadius * sphere.mRadius);
}


// This is approximate and may produce false positives, i.e. may indcate an intersection when there isn't one.
bool Intersect::intersect(const Sphere &sphere, const Frustum &frustum)
{
    // various distances
    float distance;

    // calculate our distances to each of the planes
    for (int i = 0; i < 6; ++i)
    {
        distance = frustum.mPlanes[i].signedDistance(sphere.mCentre);

        // if this distance is < -sphere.radius, we are outside
        if (distance < -sphere.mRadius)
        {
            return false;
        }
    }

    // otherwise we are fully in view
    return true;
}

bool Intersect::intersect(const AABB &aabb, const AABB &aabb2)
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

bool Intersect::intersect(const AABB &aabb, const Frustum &frustum)
{
    for (int i = 0; i < 6; ++i)
    {
        float r = fabsf(aabb.mSize.x * frustum.mPlanes[i].mNormal.x) +
                  fabsf(aabb.mSize.y * frustum.mPlanes[i].mNormal.y) +
                  fabsf(aabb.mSize.z * frustum.mPlanes[i].mNormal.z);

        // signed distance between box center and plane
        //float d = glm::dot(furstum.mPlane[i].mNormal, aabb.mCentre) + frustum.mPlanes[i].getD();
        float d = frustum.mPlanes[i].signedDistance(aabb.mCentre);

        // return signed distance
        float side = d - r;
        if (fabsf(d) < r)
        {
            side = 0.0f;
        }
        else if (d < 0.0f)
        {
            side = d + r;
        }

        if (side < 0)
        {
            return false;
        }
    }
    return true;
}


bool Intersect::intersect(const glm::vec3 &centre, const glm::vec3 &size, const Frustum &frustum)
{
    for (int i = 0; i < 6; ++i)
    {
        float r = fabsf(size.x * frustum.mPlanes[i].mNormal.x) +
                  fabsf(size.y * frustum.mPlanes[i].mNormal.y) +
                  fabsf(size.z * frustum.mPlanes[i].mNormal.z);

        // signed distance between box center and plane
        // float d = glm::dot(furstum.mPlane[i].mNormal, aabb.mCentre) + frustum.mPlanes[i].getD();
        float d = frustum.mPlanes[i].signedDistance(centre);

        // return signed distance
        float side = d - r;
        if (fabsf(d) < r)
        {
            side = 0.0f;
        }
        else if (d < 0.0f)
        {
            side = d + r;
        }

        if (side < 0)
        {
            return false;
        }
    }
    return true;
}













//bool Intersect::intersect(const AABB &aabb, const Frustum &frustum)
//{
//    for (int i = 0; i < 6; i++)
//    {
//        // maximum extent in direction of plane normal
//        float r = fabsf(aabb.mSize.x * frustum.mPlanes[i].mNormal.x) +
//                  fabsf(aabb.mSize.y * frustum.mPlanes[i].mNormal.y) +
//                  fabsf(aabb.mSize.z * frustum.mPlanes[i].mNormal.z);
//
//        // signed distance between box center and plane
//        float d = frustum.mPlanes[i].signedDistance(aabb.mCentre);
//
//        // return signed distance
//        float side = d - r;
//        if (fabsf(d) < r)
//        {
//            side = 0.0f;
//        }
//        else if (d < 0.0f)
//        {
//            side = d + r;
//        }
//
//        if (side < 0)
//        {
//            return false;
//        }
//    }
//
//    return true;
//}

//bool Intersect::intersect(const glm::vec3 &centre, const glm::vec3 &extents, const Frustum &frustum)
//{
//    for (int i = 0; i < 6; i++)
//    {
//        // maximum extent in direction of plane normal
//        float r = fabsf(0.5f * extents.x * frustum.mPlanes[i].mNormal.x) +
//                  fabsf(0.5f * extents.y * frustum.mPlanes[i].mNormal.y) +
//                  fabsf(0.5f * extents.z * frustum.mPlanes[i].mNormal.z);
//
//        // signed distance between box center and plane
//        float d = frustum.mPlanes[i].signedDistance(centre);
//
//        // return signed distance
//        float side = d - r;
//        if (fabsf(d) < r)
//        {
//            side = 0.0f;
//        }
//        else if (d < 0.0f)
//        {
//            side = d + r;
//        }
//
//        if (side < 0)
//        {
//            return false;
//        }
//    }
//
//    return true;
//}


// https://iquilezles.org/articles/frustumcorrect/
bool Intersect::intersect2(const AABB &aabb, const Frustum &frustum)
{
    glm::vec3 min = aabb.getMin();
    glm::vec3 max = aabb.getMax();

    // check box outside/inside of frustum
    for (int i = 0; i < 6; i++)
    {
        int out = 0;
        out += (frustum.mPlanes[i].signedDistance(glm::vec3(min.x, min.y, min.z)) < 0.0f) ? 1 : 0;
        out += (frustum.mPlanes[i].signedDistance(glm::vec3(max.x, min.y, min.z)) < 0.0f) ? 1 : 0;
        out += (frustum.mPlanes[i].signedDistance(glm::vec3(min.x, max.y, min.z)) < 0.0f) ? 1 : 0;
        out += (frustum.mPlanes[i].signedDistance(glm::vec3(max.x, max.y, min.z)) < 0.0f) ? 1 : 0;
        out += (frustum.mPlanes[i].signedDistance(glm::vec3(min.x, min.y, max.z)) < 0.0f) ? 1 : 0;
        out += (frustum.mPlanes[i].signedDistance(glm::vec3(max.x, min.y, max.z)) < 0.0f) ? 1 : 0;
        out += (frustum.mPlanes[i].signedDistance(glm::vec3(min.x, max.y, max.z)) < 0.0f) ? 1 : 0;
        out += (frustum.mPlanes[i].signedDistance(glm::vec3(max.x, max.y, max.z)) < 0.0f) ? 1 : 0;

        if (out == 8)
        {
            return false;
        }
    }

    // check frustum outside/inside box
    int out;
    for (int i = 0; i < 3; i++)
    {
        out = 0;
        out += frustum.mFtl[i] > max[i] ? 1 : 0;
        out += frustum.mFtr[i] > max[i] ? 1 : 0;
        out += frustum.mFbl[i] > max[i] ? 1 : 0;
        out += frustum.mFbr[i] > max[i] ? 1 : 0;
        out += frustum.mNtl[i] > max[i] ? 1 : 0;
        out += frustum.mNtr[i] > max[i] ? 1 : 0;
        out += frustum.mNbl[i] > max[i] ? 1 : 0;
        out += frustum.mNbr[i] > max[i] ? 1 : 0;

        if (out == 8)
        {
            return false;
        }

        out = 0;
        out += frustum.mFtl[i] < min[i] ? 1 : 0;
        out += frustum.mFtr[i] < min[i] ? 1 : 0;
        out += frustum.mFbl[i] < min[i] ? 1 : 0;
        out += frustum.mFbr[i] < min[i] ? 1 : 0;
        out += frustum.mNtl[i] < min[i] ? 1 : 0;
        out += frustum.mNtr[i] < min[i] ? 1 : 0;
        out += frustum.mNbl[i] < min[i] ? 1 : 0;
        out += frustum.mNbr[i] < min[i] ? 1 : 0;

        if (out == 8)
        {
            return false;
        }
    }

    return true;
}