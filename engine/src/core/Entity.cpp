#include <iostream>

#include "../../include/core/Entity.h"

using namespace PhysicsEngine;

Entity::Entity()
{
	isActive = false;

	entityId = -1;
	//globalEntityIndex = -1;

	//ind = 0;
	for(int i = 0; i < 8; i++){
		//componentTypes[i] = -1;
		//globalComponentIndices[i] = -1;
		componentIds[i] = -1;
	}	

	manager = NULL;
}

Entity::~Entity()
{

}

void Entity::setManager(Manager* manager)
{
	// if(manager->check(this)){
	// 	this->manager = manager;
	// }
	this->manager = manager;
}

// void Entity::dontDestroyOnLoad(Entity* entity)
// {	
// 	entity-
// }