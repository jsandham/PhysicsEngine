#ifndef __MESHRENDERER_DRAWER_H__
#define __MESHRENDERER_DRAWER_H__

#include "InspectorDrawer.h"

namespace PhysicsEditor
{
	class MeshRendererDrawer : public InspectorDrawer
	{
	public:
		MeshRendererDrawer();
		~MeshRendererDrawer();

		void render(World world, Guid entityId, Guid componentId);
	};
}

#endif