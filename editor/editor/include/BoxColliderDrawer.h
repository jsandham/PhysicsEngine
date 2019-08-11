#ifndef __BOXCOLLIDER_DRAWER_H__
#define __BOXCOLLIDER_DRAWER_H__

#include "InspectorDrawer.h"

namespace PhysicsEditor
{
	class BoxColliderDrawer : public InspectorDrawer
	{
		public:
			BoxColliderDrawer();
			~BoxColliderDrawer();

			void render(Component* component);
	};
}

#endif