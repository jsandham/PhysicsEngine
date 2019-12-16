#ifndef __MATERIAL_DRAWER_H__
#define __MATERIAL_DRAWER_H__

#include "InspectorDrawer.h"

namespace PhysicsEditor
{
	class MaterialDrawer : public InspectorDrawer
	{
	public:
		MaterialDrawer();
		~MaterialDrawer();

		void render(World* world, EditorClipboard& clipboard, Guid id);
	};
}

#endif