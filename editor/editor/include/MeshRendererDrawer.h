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

		void render(Component* component);
	};
}

#endif