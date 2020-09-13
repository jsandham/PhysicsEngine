#include "../../include/core/Circle.h"

using namespace PhysicsEngine;


Circle::Circle() : mCentre(glm::vec3(0.0f, 0.0f, 0.0f)), 
				   mNormal(glm::vec3(1.0f, 0.0f, 0.0f)), 
				   mRadius(1.0f)
{

}

Circle::Circle(glm::vec3 centre, glm::vec3 normal, float radius) : mCentre(centre),
																   mNormal(normal), 
																   mRadius(radius)
{
	
}

Circle::~Circle()
{

}

float Circle::getArea() const
{
	return glm::pi<float>() * mRadius * mRadius;
}

float Circle::getCircumference() const
{
	return 2.0f * glm::pi<float>() * mRadius;
}