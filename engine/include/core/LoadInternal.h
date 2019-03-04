#ifndef __LOADINTERNAL_H__
#define __LOADINTERNAL_H__

#include <vector>

#include "Asset.h"
#include "Entity.h"
#include "PoolAllocator.h"
#include "../components/Component.h"
#include "../systems/System.h"

namespace PhysicsEngine
{
	template<typename T>
	static T* create()
	{
		return getAllocator<T>().construct();
	}

	template<typename T>
	static T* create(std::vector<char> data)
	{
		return getAllocator<T>().construct(data);
	}

	Asset* loadInternalAsset(std::vector<char> data, int* index);
	Entity* loadInternalEntity(std::vector<char> data, int* index);
	Component* loadInternalComponent(std::vector<char> data, int* index, int* instanceType);
	System* loadInternalSystem(std::vector<char> data, int* index);
}

#endif