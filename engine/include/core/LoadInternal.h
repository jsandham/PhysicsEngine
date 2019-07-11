#ifndef __LOADINTERNAL_H__
#define __LOADINTERNAL_H__

#include <vector>

#include "common.h"

#include "PoolAllocator.h"
#include "Asset.h"
#include "Entity.h"
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

	template<typename T>
	static T* destroy(int index)
	{	
		return getAllocator<T>().destruct(index);
	}

	Asset* loadInternalAsset(std::vector<char> data, int type, int* index);
	Entity* loadInternalEntity(std::vector<char> data, int* index);
	Component* loadInternalComponent(std::vector<char> data, int type, int* index, itype* instanceType);
	System* loadInternalSystem(std::vector<char> data, int type, int* index);

	Entity* destroyInternalEntity(int index);
	Component* destroyInternalComponent(itype instanceType, int index);
}

#endif