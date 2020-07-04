#ifndef __INTERNAL_MATERIALS_H__
#define __INTERNAL_MATERIALS_H__

#include <string>

#include "Material.h"
#include "Guid.h"

namespace PhysicsEngine
{
	class InternalMaterials
	{
		public:
			static const Guid simpleLitMaterialId;
			static const Guid colorMaterialId;

			static Guid loadSimpleLitMaterial(World* world, const Guid shaderId);
			static Guid loadColorMaterial(World* world, const Guid shaderId);

		private:
			static Guid loadInternalMaterial(World* world, const Guid materialId, const Guid shaderId);
	};
}

#endif