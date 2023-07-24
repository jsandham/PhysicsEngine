#include <algorithm>
#include <iostream>

#include "../../include/core/SerializationEnums.h"
#include "../../include/core/SerializationYaml.h"
#include "../../include/core/Entity.h"
#include "../../include/core/World.h"

using namespace PhysicsEngine;

Entity::Entity(World *world, const Id &id) : mWorld(world), mGuid(Guid::INVALID), mId(id), mHide(HideFlag::None)
{
    mName = "Unnamed Entity";
    mDoNotDestroy = false;
}

Entity::Entity(World *world, const Guid &guid, const Id &id) : mWorld(world), mGuid(guid), mId(id), mHide(HideFlag::None)
{
    mName = "Unnamed Entity";
    mDoNotDestroy = false;
}

Entity::~Entity()
{
}

void Entity::serialize(YAML::Node &out) const
{
    out["type"] = getType();
    out["hide"] = mHide;
    out["id"] = mGuid;

    out["doNotDestroy"] = mDoNotDestroy;
    out["name"] = mName;
}

void Entity::deserialize(const YAML::Node &in)
{
    mHide = YAML::getValue<HideFlag>(in, "hide");
    mGuid = YAML::getValue<Guid>(in, "id");

    mDoNotDestroy = YAML::getValue<bool>(in, "doNotDestroy");
    mName = YAML::getValue<std::string>(in, "name");
}

int Entity::getType() const
{
    return PhysicsEngine::ENTITY_TYPE;
}

std::string Entity::getObjectName() const
{
    return PhysicsEngine::ENTITY_NAME;
}

Guid Entity::getGuid() const
{
    return mGuid;
}

Id Entity::getId() const
{
    return mId;
}

void Entity::latentDestroy()
{
    mWorld->getActiveScene()->latentDestroyEntity(getGuid());
}

void Entity::immediateDestroy()
{
    mWorld->getActiveScene()->immediateDestroyEntity(getGuid());
}

std::vector<std::pair<Guid, int>> Entity::getComponentsOnEntity() const
{
    return mWorld->getActiveScene()->getComponentsOnEntity(getGuid());
}