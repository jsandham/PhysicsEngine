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