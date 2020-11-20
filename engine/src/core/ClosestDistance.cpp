#include "../../include/core/ClosestDistance.h"
#include "../../include/core/PolynomialRoots.h"

#include "../../include/glm/glm.hpp"

#include "../../include/core/Log.h"
#include <array>
#include <string>

using namespace PhysicsEngine;

// L1 = p1 + t1 * v1
// L2 = p2 + t2 * v2
//
// d^2 = (L1 - L2)^2 = (p2 + t2*v2 - p1 - t1*v1)^2
//
// Taking second derivative w.r.t t1 and t2:
//
// d^2d
// ----- = 2 * (dot(v1, (p1 - p2)) - t2 * dot(v1, v2) + t1 * dot(v1, v1)) = 0
// dt1^2
//
// d^2d
// ----- = 2 * (dot(v2, (p2 - p1)) - t1 * dot(v1, v2) + t2 * dot(v2, v2)) = 0
// dt2^2
float ClosestDistance::closestDistance(const Ray &ray1, const Ray &ray2, float &t1, float &t2)
{
    glm::vec3 dp = ray2.mOrigin - ray1.mOrigin;

    float v1v1 = glm::dot(ray1.mDirection, ray1.mDirection);
    float v2v2 = glm::dot(ray2.mDirection, ray2.mDirection);
    float v1v2 = glm::dot(ray1.mDirection, ray2.mDirection);

    float det = v1v1 * v2v2 - v1v2 * v1v2;

    if (std::abs(det) > FLT_MIN)
    {
        float invDet = 1.0f / det;

        float dpv1 = glm::dot(dp, ray1.mDirection);
        float dpv2 = glm::dot(dp, ray2.mDirection);

        t1 = invDet * (v2v2 * dpv1 - v1v2 * dpv2);
        t2 = invDet * (v1v2 * dpv1 - v1v1 * dpv2);

        return glm::length(dp + ray2.mDirection * t2 - ray1.mDirection * t1);
    }
    else
    {
        t1 = std::numeric_limits<float>().max();
        t2 = std::numeric_limits<float>().max();

        glm::vec3 a = glm::cross(dp, ray1.mDirection);
        return std::sqrt(glm::dot(a, a) / v1v1);
    }
}

float ClosestDistance::closestDistance(const Ray &ray, const Circle &circle, float &t, glm::vec3 &circlePoint)
{
    float D = glm::dot(circle.mNormal, circle.mCentre);

    t = (D - glm::dot(circle.mNormal, ray.mOrigin)) / glm::dot(circle.mNormal, ray.mDirection);

    if (t >= 0.0f)
    { // ray intersects circle plane
        glm::vec3 planePoint = ray.getPoint(t);
        circlePoint = circle.mCentre + circle.mRadius * glm::normalize(planePoint - circle.mCentre);

        return glm::distance(planePoint, circlePoint);
    }
    else
    { // ray does not intersect circle plane
        // find closest circle point to ray origin
        glm::vec3 temp = ray.mOrigin - circle.mCentre;
        circlePoint =
            temp - (glm::dot(temp, circle.mNormal) / glm::dot(circle.mNormal, circle.mNormal)) * circle.mNormal;

        return glm::distance(ray.mOrigin, circlePoint);
    }
}

float ClosestDistance::closestDistance(const Ray &ray, const Sphere &sphere, float &t, glm::vec3 &spherePoint)
{
    glm::vec3 diff = sphere.mCentre - ray.mOrigin;

    t = glm::dot(ray.mDirection, diff) / glm::dot(ray.mDirection, ray.mDirection);

    glm::vec3 temp = ray.mOrigin + t * ray.mDirection - sphere.mCentre;
    float d = glm::sqrt(glm::dot(temp, temp)) - sphere.mRadius;

    float ratio = sphere.mRadius / (sphere.mRadius + d);

    spherePoint = sphere.mCentre + ratio * temp;

    return d;
}

// Equation for the plane that the circle lies on:
// nx * x + ny * y + nz * z = dot(N, C) where N = (nx, ny, nz) is the
// plane normal and C is the circle origin.
//
// Equation for a ray is P(t) = B + t* M.
//
// Given these equations, closest distance between a ray and a circle in 3D
// can be solved by first finding the real roots of the 4th degree polynomial:
// H(t) = (|NxM|^2*t^2 + 2 * dot((NxM), (NxD))*t + |NxD|^2)*(t + dot(M, D))^2
//      - r^2 * (|NxM|^2*t + dot((NxM), (NxD)))^2
//
// This can be simplified to:
//
// H(t) = (a*t^2 + 2*b*t + c)*(t + d)^2
//        - r^2*(a*t + b)^2
//
// where,
// a = |NxM|^2
// b = dot((NxM), (NxD))
// c = |NxD|^2
// d = dot(M, D)
//
// Which can then be further simplified to the quartic polynomial
//
// H(t) = h0 + h1*t + h2*t^2 + h3*t^3 + h4*t^4
//
// where,
// h0 = c * d^2 - b^2 * r^2
// h1 = 2 * (c * d + b * d^2 - a * b * r^2)
// h2 = c + 4 * b * d + a * d^2 - a^2 * r^2
// h3 = 2 * (b + a * d)
// h4 = a
//
// See https://www.geometrictools.com/Documentation/DistanceToCircle3.pdf
// float ClosestDistance::closestDistance(const Ray &ray, const Circle &circle, float &t)
//{
//    glm::vec3 D = ray.mOrigin - circle.mCentre;
//    glm::vec3 NxM = glm::cross(circle.mNormal, ray.mDirection);
//    glm::vec3 NxD = glm::cross(circle.mNormal, D);
//
//    if (NxM != glm::vec3(0.0f, 0.0f, 0.0f))
//    {
//        if (NxD != glm::vec3(0.0f, 0.0f, 0.0f))
//        {
//            float NdM = glm::dot(circle.mNormal, ray.mDirection);
//            if (NdM != 0.0f)
//            {
//                // H(t) = (a*t^2 + 2*b*t + c)*(t + d)^2
//                //        - r^2*(a*t + b)^2
//                //      = h0 + h1*t + h2*t^2 + h3*t^3 + h4*t^4
//                float a = glm::dot(NxM, NxM);
//                float b = glm::dot(NxM, NxD);
//                float c = glm::dot(NxD, NxD);
//                float d = glm::dot(ray.mDirection, D);
//                float rsqr = circle.mRadius * circle.mRadius;
//                float asqr = a * a, bsqr = b * b, dsqr = d * d;
//                float h0 = c * dsqr - bsqr * rsqr;
//                float h1 = 2.0f * (c * d + b * dsqr - a * b * rsqr);
//                float h2 = c + 4.0f * b * d + a * dsqr - asqr * rsqr;
//                float h3 = 2.0f * (b + a * d);
//                float h4 = a;
//
//                // Solve the quartic polynomial for real roots
//                // H(t) = h0 + h1*t + h2*t^2 + h3*t^3 + h4*t^4
//                std::vector<float> roots = PolynomialRoots::solveQuartic(h4, h3, h2, h1, h0);
//
//                int index = 0;
//                std::array<float, 4> candidates;
//                candidates.fill(10000000.0f);
//
//                for (size_t i = 0; i < roots.size(); i++)
//                {
//                    glm::vec3 lineClosest;
//                    glm::vec3 circleClosest;
//
//                    glm::vec3 NxDelta = NxD + roots[i] * NxM;
//                    if (NxDelta != glm::vec3(0.0f, 0.0f, 0.0f))
//                    {
//                        glm::vec3 delta = D + roots[i] * ray.mDirection;
//                        lineClosest = circle.mCentre + delta;
//                        delta -= glm::dot(circle.mNormal, delta) * circle.mNormal;
//                        glm::normalize(delta);
//                        circleClosest = circle.mCentre + circle.mRadius * delta;
//                    }
//                    else
//                    {
//                    }
//
//                    glm::vec3 diff = lineClosest - circleClosest;
//                    candidates[index] = glm::dot(diff, diff);
//                    index++;
//                }
//
//                std::sort(candidates.begin(), candidates.begin() + index);
//
//                for (int i = 0; i < 4; i++)
//                {
//                    std::string test3 = "candidates: " + std::to_string(candidates[i]) + "\n";
//                    Log::info(test3.c_str());
//                }
//                Log::info("\n");
//            }
//            else
//            {
//                // The line is parallel to the plane of the circle.
//                // The polynomial has the form
//                // H(t) = (t+v)^2*[(t+v)^2-(r^2-u^2)].
//                float u = glm::dot(NxM, D);
//                float v = glm::dot(ray.mDirection, D);
//                float discr = circle.mRadius * circle.mRadius - u * u;
//                if (discr > 0.0f)
//                {
//                    float rootDiscr = glm::sqrt(discr);
//                    float root0 = -v + rootDiscr;
//
//                    float root1 = -v - rootDiscr;
//                }
//                else
//                {
//                }
//            }
//        }
//    }
//    else
//    {
//        if (NxD != glm::vec3(0.0f, 0.0f, 0.0f))
//        {
//            // H(t) = |Cross(N,D)|^2*(t + Dot(M,D))^2.
//            float root0 = -glm::dot(ray.mDirection, D);
//
//            /*glm::vec3 delta = D + root0 * ray.mDirection;
//            lineClosest = circle.mCentre + delta;
//            delta -= glm::dot(circle.mNormal, delta) * circle.mNormal;
//            glm::normalize(delta);
//            circleClosest = circle.mCentre + circle.mRadius * delta;*/
//        }
//        else
//        {
//        }
//    }
//
//    return 1.0f;
//}