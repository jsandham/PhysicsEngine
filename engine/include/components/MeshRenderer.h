#ifndef __MESHRENDERER_H__
#define __MESHRENDERER_H__

#include "Component.h"

namespace PhysicsEngine
{
	class MeshRenderer : public Component
	{
		public:
			int meshId;
			int materialId;
			int meshGlobalIndex;
			int materialGlobalIndex;

		public:
			MeshRenderer();
			~MeshRenderer();
	};
}

#endif