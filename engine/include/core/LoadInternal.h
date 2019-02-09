#ifndef __LOADINTERNAL_H__
#define __LOADINTERNAL_H__

#include <vector>

#include "Asset.h"
#include "Entity.h"
#include "../components/Component.h"
#include "../systems/System.h"

namespace PhysicsEngine
{
	Asset* loadInternalAsset(std::vector<char> data);
	Entity* loadInternalEntity(std::vector<char> data);
	Component* loadInternalComponent(std::vector<char> data);
	System* loadInternalSystem(std::vector<char> data);
}

#endif