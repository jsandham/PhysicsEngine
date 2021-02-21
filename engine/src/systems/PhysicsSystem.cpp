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