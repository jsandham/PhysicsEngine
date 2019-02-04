#include "../../include/components/Rigidbody.h"

#include "../../include/core/PoolAllocator.h"

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

Rigidbody::Rigidbody(unsigned char* data)
{
	
}

Rigidbody::~Rigidbody()
{

}

void* Rigidbody::operator new(size_t size)
{
	return getAllocator<Rigidbody>().allocate();
}

void Rigidbody::operator delete(void*)
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