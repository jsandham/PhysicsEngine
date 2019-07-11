#include <iostream>
#include "../../include/components/Rigidbody.h"

#include "../../include/core/PoolAllocator.h"

using namespace PhysicsEngine;


Rigidbody::Rigidbody()
{
	std::cout << "Rigidbody default constructor called" << std::endl;

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
	deserialize(data);
}

Rigidbody::~Rigidbody()
{

}

std::vector<char> Rigidbody::serialize()
{
	RigidbodyHeader header;
	header.componentId = componentId;
	header.entityId = entityId;
	header.useGravity = useGravity;
	header.mass = mass;
	header.drag = drag;
	header.angularDrag = angularDrag;

	header.velocity = velocity;
	header.angularVelocity = angularVelocity;
	header.centreOfMass = centreOfMass;
	header.inertiaTensor = inertiaTensor;
	
	header.halfVelocity = halfVelocity;

	int numberOfBytes = sizeof(RigidbodyHeader);

	std::vector<char> data(numberOfBytes);

	memcpy(&data[0], &header, sizeof(RigidbodyHeader));

	return data;
}

void Rigidbody::deserialize(std::vector<char> data)
{
	RigidbodyHeader* header = reinterpret_cast<RigidbodyHeader*>(&data[0]);

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