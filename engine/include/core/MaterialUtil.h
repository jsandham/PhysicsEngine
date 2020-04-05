#ifndef __MATERIAL_UTIL_H__
#define __MATERIAL_UTIL_H__

#include "World.h"
#include "Material.h"

namespace PhysicsEngine
{
	class MaterialUtil
	{
		public:
			static void copyMaterialTo(World* srcWorld, Material* srcMat, World* destWorld, Material* destMat);
	};
}

#endif
