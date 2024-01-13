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

Triangle::~Triangle()
{
}

glm::vec3 Triangle::getBarycentric(const glm::vec3 &p)
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