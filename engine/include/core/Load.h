#ifndef __LOAD_H__
#define __LOAD_H__

#include "Asset.h"
#include "../components/Component.h"
#include "../systems/System.h"

namespace PhysicsEngine
{
	// load external asset defined by the user (assets with type 10 or greater)
	Asset* loadAsset(unsigned char* data);

	// load external components defined by the user (components with type 20 or greater)
	Component* loadComponent(unsigned char* data);

	// load external systems defined by the user (systems with type 10 or greater)
	System* loadSystem(unsigned char* data);
}

#endif