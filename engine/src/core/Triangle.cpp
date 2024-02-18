#include "../../include/core/Triangle.h"

using namespace PhysicsEngine;

Triangle::Triangle()
{
    mV0 = glm::vec3(0.0f, 0.0f, 0.0f);
    mV1 = glm::vec3(0.0f, 0.0f, 0.0f);
    mV2 = glm::vec3(0.0f, 0.0f, 0.0f);
}

Triangle::Triangle(const glm::vec3 &v0, const glm::vec3 &v1, const glm::vec3 &v2) : mV0(v0), mV1(v1), mV2(v2)
{
}

glm::vec3 Triangle::getBarycentric(const glm::vec3 &p) const
{
    glm::vec3 e1 = mV1 - mV0;
    glm::vec3 e2 = mV2 - mV0;

    glm::vec3 v2_ = p - mV0;
    float d00 = glm::dot(e1, e1);
    float d01 = glm::dot(e1, e2);
    float d11 = glm::dot(e2, e2);
    float d20 = glm::dot(v2_, e1);
    float d21 = glm::dot(v2_, e2);
    float d = d00 * d11 - d01 * d01;
    float v = (d11 * d20 - d01 * d21) / d;
    float w = (d00 * d21 - d01 * d20) / d;
    float u = 1 - v - w;
    return glm::vec3(u, v, w);
}

glm::vec3 Triangle::getCentroid() const
{
    return (mV0 + mV1 + mV2) / 3.0f;
}

glm::vec3 Triangle::getNormal() const
{
    //glm::vec3 p = mV1 - mV0;
    //glm::vec3 q = mV2 - mV0;

    // Calculate p vector
    float px = mV1.x - mV0.x;
    float py = mV1.y - mV0.y;
    float pz = mV1.z - mV0.z;

    // Calculate q vector
    float qx = mV2.x - mV0.x;
    float qy = mV2.y - mV0.y;
    float qz = mV2.z - mV0.z;

    // Calculate normal (p x q)
    // i  j  k
    // px py pz
    // qx qy qz
    float nx = py * qz - pz * qy;
    float ny = pz * qx - px * qz;
    float nz = px * qy - py * qx;

    return glm::vec3(nx, ny, nz);
}

glm::vec3 Triangle::getUnitNormal() const
{
    return glm::normalize(getNormal());
}

AABB Triangle::getAABBBounds() const
{
    glm::vec3 bmin = glm::min(mV0, glm::min(mV1, mV2));
    glm::vec3 bmax = glm::max(mV0, glm::max(mV1, mV2));

    AABB aabb;
    aabb.mSize = bmax - bmin;
    aabb.mCentre = bmin + 0.5f * aabb.mSize;

    return aabb;
}