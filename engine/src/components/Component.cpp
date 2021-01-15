#include <iostream>

#include "../../include/components/Component.h"
#include "../../include/core/World.h"

using namespace PhysicsEngine;

Component::Component() : Object()
{
    mEntityId = Guid::INVALID;
}

Component::Component(Guid id) : Object(id)
{
    mEntityId = Guid::INVALID;
}

Component::~Component()
{
}

Entity *Component::getEntity(World *world) const
{
    return world->getEntityById(mEntityId);
}

Guid Component::getEntityId() const
{
    return mEntityId;
}

bool Component::isInternal(int type)
{
    return type >= PhysicsEngine::MIN_INTERNAL_COMPONENT && type <= PhysicsEngine::MAX_INTERNAL_COMPONENT;
}