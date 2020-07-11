#ifndef __LOAD_H__
#define __LOAD_H__

#include <vector>
#include <unordered_map>

#include "Allocator.h"
#include "Asset.h"
#include "../components/Component.h"
#include "../systems/System.h"

namespace PhysicsEngine
{
	// get external asset defined by the user (assets with type 10 or greater)
	Asset* getAsset(std::unordered_map<int, Allocator*>* allocatorMap, int type, int index);

	// get external components defined by the user (components with type 20 or greater)
	Component* getComponent(std::unordered_map<int, Allocator*>* allocatorMap, int type, int index);

	// get external systems defined by the user (systems with type 10 or greater)
	System* getSystem(std::unordered_map<int, Allocator*>* allocatorMap, int type, int index);

	// load external asset defined by the user (assets with type 10 or greater)
	Asset* loadAsset(std::unordered_map<int, Allocator*>* allocatorMap, const std::vector<char>& data, int type, int* index);

	// load external components defined by the user (components with type 20 or greater)
	Component* loadComponent(std::unordered_map<int, Allocator*>* allocatorMap, const std::vector<char>& data, int type, int* index);

	// load external systems defined by the user (systems with type 10 or greater)
	System* loadSystem(std::unordered_map<int, Allocator*>* allocatorMap, const std::vector<char>& data, int type, int* index);

	// destroy external components defined by the user 
	Component* destroyComponent(std::unordered_map<int, Allocator*>* allocatorMap, int type, int index);
}

#endif