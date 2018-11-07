#ifndef __MESHRENDERER_H__
#define __MESHRENDERER_H__

#include "Component.h"
#include "../core/Guid.h"

namespace PhysicsEngine
{
	class MeshRenderer : public Component
	{
		public:
			Guid meshId;
			Guid materialId;

		public:
			MeshRenderer();
			~MeshRenderer();
	};
}

#endif