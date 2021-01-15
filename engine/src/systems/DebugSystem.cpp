#include <iostream>

#include "../../include/systems/DebugSystem.h"

#include "../../include/core/Input.h"
#include "../../include/core/PoolAllocator.h"
#include "../../include/core/World.h"

#include "../../include/components/MeshRenderer.h"
#include "../../include/components/Transform.h"

#include "../../include/graphics/Graphics.h"

#include "../../include/glm/glm.hpp"
#include "../../include/glm/gtc/type_ptr.hpp"

using namespace PhysicsEngine;

DebugSystem::DebugSystem() : System()
{
}

DebugSystem::DebugSystem(Guid id) : System(id)
{
  
}

DebugSystem::~DebugSystem()
{
}

std::vector<char> DebugSystem::serialize() const
{
    return serialize(mId);
}

std::vector<char> DebugSystem::serialize(const Guid &systemId) const
{
    DebugSystemHeader header;
    header.mSystemId = systemId;
    header.mUpdateOrder = static_cast<int32_t>(mOrder);

    std::vector<char> data(sizeof(DebugSystemHeader));

    memcpy(&data[0], &header, sizeof(DebugSystemHeader));

    return data;
}

void DebugSystem::deserialize(const std::vector<char> &data)
{
    const DebugSystemHeader *header = reinterpret_cast<const DebugSystemHeader *>(&data[0]);

    mId = header->mSystemId;
    mOrder = static_cast<int>(header->mUpdateOrder);
}

void DebugSystem::init(World *world)
{
    mWorld = world;
}

void DebugSystem::update(const Input &input, const Time &time)
{
}