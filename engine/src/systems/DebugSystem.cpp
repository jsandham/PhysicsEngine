#include <iostream>

#include "../../include/systems/DebugSystem.h"

#include "../../include/core/Input.h"
#include "../../include/core/PoolAllocator.h"
#include "../../include/core/World.h"
#include "../../include/core/Serialize.h"

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

void DebugSystem::serialize(std::ostream& out) const
{
    System::serialize(out);
}

void DebugSystem::deserialize(std::istream& in)
{
    System::deserialize(in);
}

void DebugSystem::init(World *world)
{
    mWorld = world;
}

void DebugSystem::update(const Input &input, const Time &time)
{
}