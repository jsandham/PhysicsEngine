#ifndef __CAPSULECOLLIDER_DRAWER_H__
#define __CAPSULECOLLIDER_DRAWER_H__

#include "InspectorDrawer.h"

namespace PhysicsEditor
{
	class CapsuleColliderDrawer : public InspectorDrawer
	{
		public:
			CapsuleColliderDrawer();
			~CapsuleColliderDrawer();

			void render(Component* component);
	};
}

#endif