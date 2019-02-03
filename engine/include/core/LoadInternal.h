#ifndef __LOADINTERNAL_H__
#define __LOADINTERNAL_H__

#include "Asset.h"
#include "Entity.h"
#include "../components/Component.h"
#include "../systems/System.h"

namespace PhysicsEngine
{
	Asset* loadInternalAsset(unsigned char* data);
	Entity* loadInternalEntity(unsigned char* data);
	Component* loadInternalComponent(unsigned char* data);
	System* loadInternalSystem(unsigned char* data);

}

#endif