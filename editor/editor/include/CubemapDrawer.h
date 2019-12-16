#ifndef __CUBEMAP_DRAWER_H__
#define __CUBEMAP_DRAWER_H__

#include "InspectorDrawer.h"

namespace PhysicsEditor
{
	class CubemapDrawer : public InspectorDrawer
	{
	public:
		CubemapDrawer();
		~CubemapDrawer();

		void render(World* world, EditorClipboard& clipboard, Guid id);
	};
}

#endif