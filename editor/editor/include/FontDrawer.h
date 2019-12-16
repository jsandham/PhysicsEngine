#ifndef __FONT_DRAWER_H__
#define __FONT_DRAWER_H__

#include "InspectorDrawer.h"

namespace PhysicsEditor
{
	class FontDrawer : public InspectorDrawer
	{
	public:
		FontDrawer();
		~FontDrawer();

		void render(World* world, EditorClipboard& clipboard, Guid id);
	};
}

#endif