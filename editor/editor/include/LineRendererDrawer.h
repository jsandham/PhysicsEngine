#ifndef __LINERENDERER_DRAWER_H__
#define __LINERENDERER_DRAWER_H__

#include "InspectorDrawer.h"

namespace PhysicsEditor
{
	class LineRendererDrawer : public InspectorDrawer
	{
	public:
		LineRendererDrawer();
		~LineRendererDrawer();

		void render(Component* component);
	};
}

#endif