#include "../../include/systems/System.h"

using namespace PhysicsEngine;

System::System()
{
    mOrder = -1;
    mSystemId = Guid::INVALID;
}

System::~System()
{
}

Guid System::getId() const
{
    return mSystemId;
}

int System::getOrder() const
{
    return mOrder;
}

bool System::isInternal(int type)
{
    return type >= PhysicsEngine::MIN_INTERNAL_SYSTEM && type <= PhysicsEngine::MAX_INTERNAL_SYSTEM;
}