#include <algorithm>

#include "../../include/systems/PhysicsSystem.h"

#include "../../include/core/AABB.h"
#include "../../include/core/Input.h"
#include "../../include/core/Physics.h"
#include "../../include/core/PoolAllocator.h"
#include "../../include/core/Sphere.h"
#include "../../include/core/Triangle.h"
#include "../../include/core/World.h"
#include "../../include/core/Serialize.h"

#include "../../include/components/BoxCollider.h"
#include "../../include/components/MeshCollider.h"
#include "../../include/components/SphereCollider.h"
#include "../../include/components/Transform.h"

using namespace PhysicsEngine;

PhysicsSystem::PhysicsSystem() : System()
{
}

PhysicsSystem::PhysicsSystem(Guid id) : System(id)
{
}

PhysicsSystem::~PhysicsSystem()
{
}

std::vector<char> PhysicsSystem::serialize() const
{
    return serialize(mId);
}

std::vector<char> PhysicsSystem::serialize(const Guid &systemId) const
{
    PhysicsSystemHeader header;
    header.mSystemId = systemId;
    header.mGravity = mGravity;
    header.mTimestep = mTimestep;
    header.mUpdateOrder = static_cast<int32_t>(mOrder);

    std::vector<char> data(sizeof(PhysicsSystemHeader));

    memcpy(&data[0], &header, sizeof(PhysicsSystemHeader));

    return data;
}

void PhysicsSystem::deserialize(const std::vector<char> &data)
{
    const PhysicsSystemHeader *header = reinterpret_cast<const PhysicsSystemHeader *>(&data[0]);

    mId = header->mSystemId;
    mOrder = static_cast<int>(header->mUpdateOrder);
    mGravity = header->mGravity;
    mTimestep = header->mTimestep;
}

void PhysicsSystem::serialize(std::ostream& out) const
{
    System::serialize(out);

    PhysicsEngine::write<float>(out, mGravity);
    PhysicsEngine::write<float>(out, mTimestep);
}

void PhysicsSystem::deserialize(std::istream& in)
{
    System::deserialize(in);

    PhysicsEngine::read<float>(in, mGravity);
    PhysicsEngine::read<float>(in, mTimestep);
}

void PhysicsSystem::init(World *world)
{
    mWorld = world;
}

void PhysicsSystem::update(const Input &input, const Time &time)
{
}