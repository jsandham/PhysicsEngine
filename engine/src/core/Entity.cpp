#include <algorithm>
#include <iostream>

#include "../../include/core/Entity.h"
#include "../../include/core/PoolAllocator.h"
#include "../../include/core/World.h"

using namespace PhysicsEngine;

Entity::Entity()
{
    mEntityId = Guid::INVALID;
    mEntityName = "Unnamed Entity";
    mDoNotDestroy = false;
}

Entity::Entity(const std::vector<char> &data)
{
    deserialize(data);
}

Entity::~Entity()
{
}

std::vector<char> Entity::serialize() const
{
    return serialize(mEntityId);
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

    mEntityId = header->mEntityId;
    mEntityName = std::string(header->mEntityName);
    mDoNotDestroy = static_cast<bool>(header->mDoNotDestroy);
}

void Entity::latentDestroy(World *world)
{
    world->latentDestroyEntity(mEntityId);
}

void Entity::immediateDestroy(World *world)
{
    world->immediateDestroyEntity(mEntityId);
}

std::vector<std::pair<Guid, int>> Entity::getComponentsOnEntity(World *world)
{
    return world->getComponentsOnEntity(mEntityId);
}

Guid Entity::getId() const
{
    return mEntityId;
}

std::string Entity::getName() const
{
    return mEntityName;
}

void Entity::setName(const std::string &name)
{
    mEntityName = name;
}