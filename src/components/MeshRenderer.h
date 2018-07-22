#ifndef __MESHRENDERER_H__
#define __MESHRENDERER_H__

#include "Component.h"

namespace PhysicsEngine
{
	class MeshRenderer : public Component
	{
		public:
			int meshFilter;
			int materialFilter;

		public:
			MeshRenderer();
			MeshRenderer(Entity* entity);
			~MeshRenderer();
	};
}

#endif