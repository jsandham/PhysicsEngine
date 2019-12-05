#include <iostream>

#include "../../include/components/Component.h"
#include "../../include/core/World.h"

using namespace PhysicsEngine;

Component::Component()
{
	componentId = Guid::INVALID;
	entityId = Guid::INVALID;
}

Component::~Component()
{
	componentId = Guid::INVALID;
	entityId = Guid::INVALID;
}

Entity* Component::getEntity(World* world)
{
	return world->getEntity(entityId);
}