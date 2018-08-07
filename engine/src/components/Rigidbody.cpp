#include "../../include/components/Rigidbody.h"

using namespace PhysicsEngine;


Rigidbody::Rigidbody()
{
	useGravity = true;
	mass = 1.0f;
	drag = 0.0f;
	angularDrag = 0.05f;

	velocity = glm::vec3(0.0f, 0.0f, 0.0f);
	centreOfMass = glm::vec3(0.0f, 0.0f, 0.0f);
	angularVelocity = glm::vec3(0.0f, 0.0f, 0.0f);
	inertiaTensor = glm::mat3(1.0f);

	halfVelocity = glm::vec3(0.0f, 0.0f, 0.0f);
}

Rigidbody::~Rigidbody()
{

}