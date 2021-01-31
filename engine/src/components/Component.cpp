#include <iostream>

#include "../../include/components/Component.h"
#include "../../include/core/World.h"
#include "../../include/core/Serialize.h"

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

void Component::serialize(std::ostream& out) const
{
    Object::serialize(out);

    PhysicsEngine::write<Guid>(out, mEntityId);
}

void Component::deserialize(std::istream& in)
{
    Object::deserialize(in);

    PhysicsEngine::read<Guid>(in, mEntityId);
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