#include <iostream>

#include "../../include/components/Component.h"
#include "../../include/core/Manager.h"

using namespace PhysicsEngine;

Component::Component()
{
	isActive = false;

	globalEntityIndex = -1;
	globalComponentIndex = -1;
	componentId = -1;
	entityId = -1;

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
	return manager->getEntity(globalEntityIndex);
}