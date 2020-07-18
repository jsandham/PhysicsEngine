#include <iostream>

#include "../../include/components/Component.h"
#include "../../include/core/World.h"

using namespace PhysicsEngine;

Component::Component()
{
	mComponentId = Guid::INVALID;
	mEntityId = Guid::INVALID;
}

Component::~Component()
{

}

Entity* Component::getEntity(World* world) const
{
	return world->getEntityById(mEntityId);
}

Guid Component::getId() const
{
	return mComponentId;
}

Guid Component::getEntityId() const
{
	return mEntityId;
}