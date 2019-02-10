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

Rigidbody::Rigidbody(std::vector<char> data)
{
	size_t index = sizeof(int);
	index += sizeof(char);
	RigidbodyHeader* header = reinterpret_cast<RigidbodyHeader*>(&data[index]);

	componentId = header->componentId;
	entityId = header->entityId;
	useGravity = header->useGravity;
	mass = header->mass;
	drag = header->drag;
	angularDrag = header->angularDrag;
	velocity = header->velocity;
	angularVelocity = header->angularVelocity;
	centreOfMass = header->centreOfMass;
	inertiaTensor = header->inertiaTensor;

	halfVelocity = header->halfVelocity;
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