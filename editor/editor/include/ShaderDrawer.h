#ifndef __SHADER_DRAWER_H__
#define __SHADER_DRAWER_H__

#include "InspectorDrawer.h"

namespace PhysicsEditor
{
	class ShaderDrawer : public InspectorDrawer
	{
		public:
			ShaderDrawer();
			~ShaderDrawer();

			void render(World* world, EditorClipboard& clipboard, Guid id);
	};
}

#endif