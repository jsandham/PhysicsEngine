#ifndef __SPHERECOLLIDER_DRAWER_H__
#define __SPHERECOLLIDER_DRAWER_H__

#include "InspectorDrawer.h"

namespace PhysicsEditor
{
	class SphereColliderDrawer : public InspectorDrawer
	{
	public:
		SphereColliderDrawer();
		~SphereColliderDrawer();

		void render(World* world, EditorClipboard& clipboard, Guid id);
	};
}

#endif