#include "../../include/systems/System.h"

using namespace PhysicsEngine;

System::System()
{
	order = -1;
	systemId = Guid::INVALID;
}

System::~System()
{

}

Guid System::getId() const
{
	return systemId;
}

int System::getOrder() const
{
	return order;
}