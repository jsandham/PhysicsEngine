#include "../../include/systems/System.h"

using namespace PhysicsEngine;

System::System() : Object()
{
    mOrder = -1;
}

System::System(Guid id) : Object(id)
{
    mOrder = -1;
}

System::~System()
{
}

int System::getOrder() const
{
    return mOrder;
}

bool System::isInternal(int type)
{
    return type >= PhysicsEngine::MIN_INTERNAL_SYSTEM && type <= PhysicsEngine::MAX_INTERNAL_SYSTEM;
}