#include <algorithm>
#include <iostream>

#include "../../include/core/Entity.h"
#include "../../include/core/PoolAllocator.h"
#include "../../include/core/World.h"
#include "../../include/core/Serialize.h"

using namespace PhysicsEngine;

Entity::Entity() : Object()
{
    mEntityName = "Unnamed Entity";
    mDoNotDestroy = false;
}

Entity::Entity(Guid id) : Object(id)
{
    mEntityName = "Unnamed Entity";
    mDoNotDestroy = false;
}

Entity::~Entity()
{
}

std::vector<char> Entity::serialize() const
{
    return serialize(mId);
}

std::vector<char> Entity::serialize(const Guid &entityId) const
{
    EntityHeader header;
    header.mEntityId = entityId;
    header.mDoNotDestroy = static_cast<uint8_t>(mDoNotDestroy);

    std::size_t len = std::min(size_t(64 - 1), mEntityName.size());
    memcpy(&header.mEntityName[0], &mEntityName[0], len);
    header.mEntityName[len] = '\0';

    std::vector<char> data(sizeof(EntityHeader));

    memcpy(&data[0], &header, sizeof(EntityHeader));

    return data;
}

void Entity::deserialize(const std::vector<char> &data)
{
    const EntityHeader *header = reinterpret_cast<const EntityHeader *>(&data[0]);

    mId = header->mEntityId;
    mEntityName = std::string(header->mEntityName);
    mDoNotDestroy = static_cast<bool>(header->mDoNotDestroy);
}

void Entity::serialize(std::ostream& out) const
{
    Object::serialize(out);

    PhysicsEngine::write<bool>(out, mDoNotDestroy);
    PhysicsEngine::write<std::string>(out, mEntityName);
}

void Entity::deserialize(std::istream& in)
{
    Object::deserialize(in);

    PhysicsEngine::read<bool>(in, mDoNotDestroy);
    PhysicsEngine::read<std::string>(in, mEntityName);
}

void Entity::latentDestroy(World *world)
{
    world->latentDestroyEntity(mId);
}

void Entity::immediateDestroy(World *world)
{
    world->immediateDestroyEntity(mId);
}

std::vector<std::pair<Guid, int>> Entity::getComponentsOnEntity(World *world)
{
    return world->getComponentsOnEntity(mId);
}

std::string Entity::getName() const
{
    return mEntityName;
}

void Entity::setName(const std::string &name)
{
    mEntityName = name;
}