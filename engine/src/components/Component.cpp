#include <iostream>

#include "../../include/components/Component.h"

using namespace PhysicsEngine;

Component::Component()
{
	globalEntityIndex = -1;
	globalComponentIndex = -1;
	componentId = -1;
	entityId = -1;
}

Component::~Component()
{
	
}

Entity* Component::getEntity(std::vector<Entity*> entities)
{
	return entities[globalEntityIndex];
}