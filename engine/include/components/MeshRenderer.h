#ifndef __MESHRENDERER_H__
#define __MESHRENDERER_H__

#include "Component.h"
#include "../core/Guid.h"

namespace PhysicsEngine
{
// #pragma pack(push, 1)
	struct MeshRendererData
	{
		Guid componentId;
		Guid entityId;
		Guid meshId;
		Guid materialId;
	};
// #pragma pack(pop)

	class MeshRenderer : public Component
	{
		public:
			Guid meshId;
			Guid materialId;

		public:
			MeshRenderer();
			~MeshRenderer();

			void load(MeshRendererData data);
	};
}

#endif