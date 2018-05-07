#include "Rigidbody.h"

using namespace PhysicsEngine;


Rigidbody::Rigidbody()
{
	mass = 1.0f;
	drag = 0.0f;
	angularDrag = 0.05f;

	centreOfMass = glm::vec3(0.0f, 0.0f, 0.0f);
	angularVelocity = glm::vec3(0.0f, 0.0f, 0.0f);
	inertiaTensor = glm::mat3(1.0f);
}

Rigidbody::Rigidbody(Entity *entity)
{
	this->entity = entity;

	mass = 1.0f;
	drag = 0.0f;
	angularDrag = 0.05f;

	centreOfMass = glm::vec3(0.0f, 0.0f, 0.0f);
	angularVelocity = glm::vec3(0.0f, 0.0f, 0.0f);
	inertiaTensor = glm::mat3(1.0f);
}

Rigidbody::~Rigidbody()
{

}

glm::vec3 Rigidbody::GetAngularVelocity()
{
	return angularVelocity;
}

glm::vec3 Rigidbody::GetCentreOfMass()
{
	return centreOfMass;
}

glm::mat3 Rigidbody::GetInertiaTensor()
{
	return inertiaTensor;
}