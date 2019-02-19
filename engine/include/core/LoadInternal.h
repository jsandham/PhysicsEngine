#ifndef __LOADINTERNAL_H__
#define __LOADINTERNAL_H__

#include <vector>

#include "Asset.h"
#include "Entity.h"
#include "../components/Component.h"
#include "../systems/System.h"

namespace PhysicsEngine
{
	Asset* loadInternalAsset(std::vector<char> data, int* index);
	Entity* loadInternalEntity(std::vector<char> data, int* index);
	Component* loadInternalComponent(std::vector<char> data, int* index, int* instanceType);
	System* loadInternalSystem(std::vector<char> data, int* index);
}

#endif