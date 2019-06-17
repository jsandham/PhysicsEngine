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
	size_t index = sizeof(int);
	index += sizeof(char);
	EntityHeader* header = reinterpret_cast<EntityHeader*>(&data[index]);

	entityId = header->entityId;
}

Entity::~Entity()
{

}

void Entity::latentDestroy(World* world)
{
	world->latentDestroyEntity(entityId);
}

void Entity::immediateDestroy(World* world)
{
	world->immediateDestroyEntity(entityId);
}