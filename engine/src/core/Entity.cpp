#include <iostream>

#include "../../include/core/Entity.h"
#include "../../include/core/Manager.h"

using namespace PhysicsEngine;

Entity::Entity()
{
	entityId = -1;
	
	for(int i = 0; i < 8; i++){
		componentIds[i] = -1;
	}	

	manager = NULL;
}

Entity::~Entity()
{

}

void Entity::setManager(Manager* manager)
{
	this->manager = manager;
}

void Entity::latentDestroy()
{
	manager->latentDestroy(entityId);
}

void Entity::immediateDestroy()
{
	manager->immediateDestroy(entityId);
}

Entity* Entity::instantiate()
{
	return manager->instantiate(entityId);
}