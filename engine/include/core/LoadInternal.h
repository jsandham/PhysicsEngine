#ifndef __LOADINTERNAL_H__
#define __LOADINTERNAL_H__

#include <vector>
#include <map>

#include "Allocator.h"
#include "PoolAllocator.h"
#include "Asset.h"
#include "Entity.h"
#include "../components/Component.h"
#include "../systems/System.h"

namespace PhysicsEngine
{
	Entity* loadInternalEntity(Allocator* allocator, std::vector<char> data, int* index);
	Component* loadInternalComponent(std::map<int, Allocator*>* allocatorMap, std::vector<char> data, int type, int* index);
	System* loadInternalSystem(std::map<int, Allocator*>* allocatorMap, std::vector<char> data, int type, int* index);
	Asset* loadInternalAsset(std::map<int, Allocator*>* allocatorMap, std::vector<char> data, int type, int* index);

	Entity* destroyInternalEntity(Allocator* allocator, int index);
	Component* destroyInternalComponent(std::map<int, Allocator*>* allocatorMap, int type, int index);
}

#endif