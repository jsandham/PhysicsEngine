#ifndef __MESH_DRAWER_H__
#define __MESH_DRAWER_H__

#include "InspectorDrawer.h"

namespace PhysicsEditor
{
	class MeshDrawer : public InspectorDrawer
	{
	public:
		MeshDrawer();
		~MeshDrawer();

		void render(World* world, EditorClipboard& clipboard, Guid id);
	};
}

#endif