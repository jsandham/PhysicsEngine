#include <iostream>

#include "../../include/components/Component.h"
#include "../../include/core/Manager.h"

using namespace PhysicsEngine;

Component::Component()
{
	isActive = false;

	componentId = Guid::INVALID;
	entityId = Guid::INVALID;

	manager = NULL;
}

Component::~Component()
{
	
}

void Component::setManager(Manager* manager)
{
	this->manager = manager;
}

Entity* Component::getEntity()
{
	return manager->getEntity(entityId);
}