#include <iostream>

#include "../../include/components/Component.h"
#include "../../include/core/Serialization.h"
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

void Component::serialize(YAML::Node &out) const
{
    Object::serialize(out);

    out["entityId"] = mEntityId;
}

void Component::deserialize(const YAML::Node &in)
{
    Object::deserialize(in);

    mEntityId = YAML::getValue<Guid>(in, "entityId");
}

Entity *Component::getEntity(const World *world) const
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