#include <algorithm>

#include "../../include/systems/PhysicsSystem.h"

#include "../../include/core/AABB.h"
#include "../../include/core/Input.h"
#include "../../include/core/Physics.h"
#include "../../include/core/PoolAllocator.h"
#include "../../include/core/Sphere.h"
#include "../../include/core/Triangle.h"
#include "../../include/core/World.h"

#include "../../include/components/BoxCollider.h"
#include "../../include/components/MeshCollider.h"
#include "../../include/components/SphereCollider.h"
#include "../../include/components/Transform.h"

using namespace PhysicsEngine;

PhysicsSystem::PhysicsSystem(World *world, const Id &id) : System(world, id)
{
}

PhysicsSystem::PhysicsSystem(World *world, const Guid &guid, const Id &id) : System(world, guid, id)
{
}

PhysicsSystem::~PhysicsSystem()
{
}

void PhysicsSystem::serialize(YAML::Node &out) const
{
    System::serialize(out);

    out["gravity"] = mGravity;
    out["timestep"] = mTimestep;
}

void PhysicsSystem::deserialize(const YAML::Node &in)
{
    System::deserialize(in);

    mGravity = YAML::getValue<float>(in, "gravity");
    mTimestep = YAML::getValue<float>(in, "timestep");
}

int PhysicsSystem::getType() const
{
    return PhysicsEngine::PHYSICSSYSTEM_TYPE;
}

std::string PhysicsSystem::getObjectName() const
{
    return PhysicsEngine::PHYSICSSYSTEM_NAME;
}

void PhysicsSystem::init(World *world)
{
    mWorld = world;
}

void PhysicsSystem::update(const Input &input, const Time &time)
{
}