#ifndef __RIGIDBODY_DRAWER_H__
#define __RIGIDBODY_DRAWER_H__

#include "InspectorDrawer.h"

namespace PhysicsEditor
{
	class RigidbodyDrawer : public InspectorDrawer
	{
		public:
			RigidbodyDrawer();
			~RigidbodyDrawer();

			void render(World* world, EditorProject& project, EditorScene& scene, EditorClipboard& clipboard, Guid id);
	};
}

#endif