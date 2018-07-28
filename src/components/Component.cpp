#include <iostream>
#include "Component.h"

using namespace PhysicsEngine;

Component::Component()
{
	
}

Component::~Component()
{
	
}

Entity* Component::getEntity(std::vector<Entity*> entities)
{
	return entities[globalEntityIndex];
}