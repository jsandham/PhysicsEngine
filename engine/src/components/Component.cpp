#include <iostream>

#include "../../include/components/Component.h"
#include "../../include/core/World.h"

using namespace PhysicsEngine;

Component::Component()
{
	componentId = Guid::INVALID;
	entityId = Guid::INVALID;

	world = NULL;
}

Component::~Component()
{
	
}

void Component::setManager(World* world)
{
	this->world = world;
}

Entity* Component::getEntity()
{
	return world->getEntity(entityId);
}