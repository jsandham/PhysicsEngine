#ifndef __LOADINTERNALSYSTEM_H__
#define __LOADINTERNALSYSTEM_H__

#include "System.h"

namespace PhysicsEngine
{
	// load internal systems defined by the engine (systems with type 0-9)
	System* loadInternalSystem(unsigned char* data);
}

#endif