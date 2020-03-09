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

}

Entity* Component::getEntity(World* world) const
{
	return world->getEntity(entityId);
}

Guid Component::getId() const
{
	return componentId;
}

Guid Component::getEntityId() const
{
	return entityId;
}