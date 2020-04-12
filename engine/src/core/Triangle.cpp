#include "../../include/core/Triangle.h"

using namespace PhysicsEngine;

Triangle::Triangle()
{
	v0 = glm::vec3(0.0f, 0.0f, 0.0f);
	v1 = glm::vec3(0.0f, 0.0f, 0.0f);
	v2 = glm::vec3(0.0f, 0.0f, 0.0f);
}

Triangle::Triangle(glm::vec3 v0, glm::vec3 v1, glm::vec3 v2)
{
	this->v0 = v0;
	this->v1 = v1;
	this->v2 = v2;
}

Triangle::~Triangle()
{

}