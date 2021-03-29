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

void System::serialize(YAML::Node &out) const
{
    Object::serialize(out);

    out["order"] = mOrder;
}

void System::deserialize(const YAML::Node &in)
{
    Object::deserialize(in);

    mOrder = YAML::getValue<size_t>(in, "order");
}

size_t System::getOrder() const
{
    return mOrder;
}

bool System::isInternal(int type)
{
    return type >= PhysicsEngine::MIN_INTERNAL_SYSTEM && type <= PhysicsEngine::MAX_INTERNAL_SYSTEM;
}