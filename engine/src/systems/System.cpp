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

int System::getOrder() const
{
	return order;
}