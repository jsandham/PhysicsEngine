#include <algorithm>
#include <iostream>

#include "../../include/core/Entity.h"
#include "../../include/core/Serialization.h"
#include "../../include/core/World.h"

using namespace PhysicsEngine;

Entity::Entity() : Object()
{
    mName = "Unnamed Entity";
    mDoNotDestroy = false;
    mHide = false;
}

Entity::Entity(Guid id) : Object(id)
{
    mName = "Unnamed Entity";
    mDoNotDestroy = false;
    mHide = false;
}

Entity::~Entity()
{
}

void Entity::serialize(std::ostream &out) const
{
    Object::serialize(out);

    PhysicsEngine::write<bool>(out, mDoNotDestroy);
    PhysicsEngine::write<bool>(out, mHide);
    PhysicsEngine::write<std::string>(out, mName);
}

void Entity::deserialize(std::istream &in)
{
    Object::deserialize(in);

    PhysicsEngine::read<bool>(in, mDoNotDestroy);
    PhysicsEngine::read<bool>(in, mHide);
    PhysicsEngine::read<std::string>(in, mName);
}

void Entity::serialize(YAML::Node& out) const
{
    Object::serialize(out);

    out["doNotDestroy"] = mDoNotDestroy;
    out["hide"] = mHide;
    out["name"] = mName;
}

void Entity::deserialize(const YAML::Node& in)
{
    Object::deserialize(in);

    mDoNotDestroy = YAML::getValue<bool>(in, "doNotDestroy");
    mHide = YAML::getValue<bool>(in, "hide");
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

void Entity::latentDestroy(World *world)
{
    world->latentDestroyEntity(mId);
}

void Entity::immediateDestroy(World *world)
{
    world->immediateDestroyEntity(mId);
}

std::vector<std::pair<Guid, int>> Entity::getComponentsOnEntity(const World *world) const
{
    return world->getComponentsOnEntity(mId);
}

std::string Entity::getName() const
{
    return mName;
}

void Entity::setName(const std::string &name)
{
    mName = name;
}