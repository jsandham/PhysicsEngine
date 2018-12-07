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

void Rigidbody::load(RigidbodyData data)
{
	entityId = data.entityId;
	componentId = data.componentId;

	useGravity = data.useGravity;
	mass = data.mass;
	drag = data.drag;
	angularDrag = data.angularDrag;

	velocity = data.velocity;
	centreOfMass = data.centreOfMass;
	angularVelocity = data.angularVelocity;
	inertiaTensor = data.inertiaTensor;

	halfVelocity = data.halfVelocity;
}