#include <iostream>

#include "../../include/core/Entity.h"

using namespace PhysicsEngine;

Entity::Entity()
{
	entityId = -1;
	globalEntityIndex = -1;

	ind = 0;
	for(int i = 0; i < 8; i++){
		componentTypes[i] = -1;
		globalComponentIndices[i] = -1;
		componentIds[i] = -1;
	}	
}

Entity::~Entity()
{

}