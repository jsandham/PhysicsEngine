#ifndef __MESHRENDERER_H__
#define __MESHRENDERER_H__

#include <vector>

#include "Component.h"

namespace PhysicsEngine
{
#pragma pack(push, 1)
	struct MeshRendererHeader
	{
		Guid componentId;
		Guid entityId;
		Guid meshId;
		Guid materialId;
	};
#pragma pack(pop)

	class MeshRenderer : public Component
	{
		public:
			Guid meshId;
			Guid materialId;

		public:
			MeshRenderer();
			MeshRenderer(std::vector<char> data);
			~MeshRenderer();

			void* operator new(size_t size);
			void operator delete(void*);
	};
}

#endif