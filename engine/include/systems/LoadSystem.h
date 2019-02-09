#ifndef __LOADSYSTEM_H__
#define __LOADSYSTEM_H__

#include <vector>

#include "System.h"

namespace PhysicsEngine
{
	// load external systems defined by the user (systems with type 10 or greater)
	System* loadSystem(std::vector<char> data);
}

#endif