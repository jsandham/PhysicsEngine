#include "../../include/systems/System.h"
#include "../../include/core/Serialization.h"

using namespace PhysicsEngine;

System::System() : Object()
{
    mOrder = 0;
}

System::System(Guid id) : Object(id)
{
    mOrder = 0;
}

System::~System()
{
}

void System::serialize(std::ostream &out) const
{
    Object::serialize(out);

    PhysicsEngine::write<size_t>(out, mOrder);
}

void System::deserialize(std::istream &in)
{
    Object::deserialize(in);

    PhysicsEngine::read<size_t>(in, mOrder);
}

size_t System::getOrder() const
{
    return mOrder;
}

bool System::isInternal(int type)
{
    return type >= PhysicsEngine::MIN_INTERNAL_SYSTEM && type <= PhysicsEngine::MAX_INTERNAL_SYSTEM;
}