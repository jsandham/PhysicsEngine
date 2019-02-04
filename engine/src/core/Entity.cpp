#include <iostream>

#include "../../include/core/PoolAllocator.h"
#include "../../include/core/Entity.h"
#include "../../include/core/World.h"

using namespace PhysicsEngine;

Entity::Entity()
{
	entityId = Guid::INVALID;
	
	world = NULL;
}

Entity::Entity(unsigned char* data)
{
	
}

Entity::~Entity()
{

}

void* Entity::operator new(size_t size)
{
	return getAllocator<Entity>().allocate();
}

void Entity::operator delete(void*)
{

}

void Entity::load(EntityData data)
{
	entityId = data.entityId;
}

void Entity::setWorld(World* world)
{
	this->world = world;
}

void Entity::latentDestroy()
{
	world->latentDestroy(entityId);
}

void Entity::immediateDestroy()
{
	world->immediateDestroy(entityId);
}

Entity* Entity::instantiate()
{
	return world->instantiate(entityId);
}