#include "../../include/components/Component.h"
#include "../../include/core/World.h"

using namespace PhysicsEngine;

Component::Component(World* world) : Object(world)
{
    mEntityId = -1;
}

Component::Component(World* world, Id id) : Object(world, id)
{
    mEntityId = -1;
}

Component::~Component()
{
}

void Component::serialize(YAML::Node &out) const
{
    Object::serialize(out);

    out["entityId"] = mWorld->getGuidOf(mEntityId);
}

void Component::deserialize(const YAML::Node &in)
{
    Object::deserialize(in);

    mEntityId = mWorld->getIdOf(YAML::getValue<Guid>(in, "entityId"));
}

Entity* Component::getEntity() const
{
    return mWorld->getEntityById(mEntityId);
}

Id Component::getEntityId() const
{
    return mEntityId;
}

bool Component::isInternal(int type)
{
    return type >= PhysicsEngine::MIN_INTERNAL_COMPONENT && type <= PhysicsEngine::MAX_INTERNAL_COMPONENT;
}