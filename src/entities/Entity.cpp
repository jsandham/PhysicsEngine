#include <iostream>
#include "Entity.h"

using namespace PhysicsEngine;

Entity::Entity()
{
	ind = 0;
	for(int i = 0; i < 8; i++){
		componentTypes[i] = -1;
		globalComponentIndices[i] = -1;
	}	
}

Entity::~Entity()
{

}