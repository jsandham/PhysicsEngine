#ifndef __LOAD_H__
#define __LOAD_H__

#include <vector>

#include "Asset.h"
#include "../components/Component.h"
#include "../systems/System.h"

namespace PhysicsEngine
{
	// load external asset defined by the user (assets with type 10 or greater)
	Asset* loadAsset(std::vector<char> data, int* index);

	// load external components defined by the user (components with type 20 or greater)
	Component* loadComponent(std::vector<char> data, int* index);

	// load external systems defined by the user (systems with type 10 or greater)
	System* loadSystem(std::vector<char> data, int* index);
}

#endif