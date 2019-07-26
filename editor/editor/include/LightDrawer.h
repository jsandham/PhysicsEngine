#ifndef __LIGHT_DRAWER_H__
#define __LIGHT_DRAWER_H__

#include "InspectorDrawer.h"

namespace PhysicsEditor
{
	class LightDrawer : public InspectorDrawer
	{
		public:
			LightDrawer();
			~LightDrawer();

			void render(Component* component);
	};
}

#endif