#include "../../include/core/Triangle.h"

using namespace PhysicsEngine;

Triangle::Triangle()
{
    mV0 = glm::vec3(0.0f, 0.0f, 0.0f);
    mV1 = glm::vec3(0.0f, 0.0f, 0.0f);
    mV2 = glm::vec3(0.0f, 0.0f, 0.0f);
}

Triangle::Triangle(glm::vec3 v0, glm::vec3 v1, glm::vec3 v2)
{
    this->mV0 = v0;
    this->mV1 = v1;
    this->mV2 = v2;
}

Triangle::~Triangle()
{
}