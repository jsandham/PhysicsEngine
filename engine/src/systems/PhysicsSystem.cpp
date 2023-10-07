#include <algorithm>

#include "../../include/systems/PhysicsSystem.h"

#include "../../include/core/SerializationYaml.h"
#include "../../include/core/AABB.h"
#include "../../include/core/Input.h"
#include "../../include/core/Physics.h"
#include "../../include/core/PoolAllocator.h"
#include "../../include/core/Sphere.h"
#include "../../include/core/Triangle.h"
#include "../../include/core/World.h"

#include "../../include/components/BoxCollider.h"
#include "../../include/components/SphereCollider.h"
#include "../../include/components/Transform.h"

using namespace PhysicsEngine;

PhysicsSystem::PhysicsSystem(World *world, const Id &id) : mWorld(world), mGuid(Guid::INVALID), mId(id), mHide(HideFlag::None)
{
    mEnabled = true;
}

PhysicsSystem::PhysicsSystem(World *world, const Guid &guid, const Id &id) : mWorld(world), mGuid(guid), mId(id), mHide(HideFlag::None)
{
    mEnabled = true;
}

PhysicsSystem::~PhysicsSystem()
{
}

void PhysicsSystem::serialize(YAML::Node &out) const
{
    out["type"] = getType();
    out["hide"] = mHide;
    out["id"] = mGuid;

    out["gravity"] = mGravity;
    out["timestep"] = mTimestep;
}

void PhysicsSystem::deserialize(const YAML::Node &in)
{
    mHide = YAML::getValue<HideFlag>(in, "hide");
    mGuid = YAML::getValue<Guid>(in, "id");

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

Guid PhysicsSystem::getGuid() const
{
    return mGuid;
}

Id PhysicsSystem::getId() const
{
    return mId;
}

void PhysicsSystem::init(World *world)
{
    mWorld = world;
}

void PhysicsSystem::update(const Input &input, const Time &time)
{
}