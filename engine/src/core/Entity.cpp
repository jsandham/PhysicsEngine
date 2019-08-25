#include <iostream>

#include "../../include/core/PoolAllocator.h"
#include "../../include/core/Entity.h"
#include "../../include/core/World.h"

using namespace PhysicsEngine;

Entity::Entity()
{
	entityId = Guid::INVALID;
}

Entity::Entity(std::vector<char> data)
{
	deserialize(data);
}

Entity::~Entity()
{

}

std::vector<char> Entity::serialize()
{
	EntityHeader header;
	header.entityId = entityId;

	int numberOfBytes = sizeof(EntityHeader);

	std::vector<char> data(numberOfBytes);

	memcpy(&data[0], &header, sizeof(EntityHeader));

	return data;
}

void Entity::deserialize(std::vector<char> data)
{
	EntityHeader* header = reinterpret_cast<EntityHeader*>(&data[0]);

	entityId = header->entityId;
}

void Entity::latentDestroy(World* world)
{
	world->latentDestroyEntity(entityId);
}

void Entity::immediateDestroy(World* world)
{
	world->immediateDestroyEntity(entityId);
}

std::vector<std::pair<Guid, int>> Entity::getComponentsOnEntity(World* world)
{
	return world->getComponentsOnEntity(entityId);
}