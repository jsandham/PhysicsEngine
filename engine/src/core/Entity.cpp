#include <algorithm>
#include <iostream>

#include "../../include/core/Entity.h"
#include "../../include/core/Serialization.h"
#include "../../include/core/World.h"

using namespace PhysicsEngine;

Entity::Entity(World *world) : Object(world)
{
    mName = "Unnamed Entity";
    mDoNotDestroy = false;
}

Entity::Entity(World *world, const Guid& id) : Object(world, id)
{
    mName = "Unnamed Entity";
    mDoNotDestroy = false;
}

Entity::~Entity()
{
}

void Entity::serialize(YAML::Node &out) const
{
    Object::serialize(out);

    out["doNotDestroy"] = mDoNotDestroy;
    out["name"] = mName;
}

void Entity::deserialize(const YAML::Node &in)
{
    Object::deserialize(in);

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

void Entity::latentDestroy()
{
    mWorld->latentDestroyEntity(getId());
}

void Entity::immediateDestroy()
{
    mWorld->immediateDestroyEntity(getId());
}

std::vector<std::pair<Guid, int>> Entity::getComponentsOnEntity() const
{
    return mWorld->getComponentsOnEntity(getId());
}

std::string Entity::getName() const
{
    return mName;
}

void Entity::setName(const std::string &name)
{
    mName = name;
}