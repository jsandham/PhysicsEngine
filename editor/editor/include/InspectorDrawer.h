#ifndef __INSPECTOR_DRAWER_H__
#define __INSPECTOR_DRAWER_H__

#include "components/Component.h"

using namespace PhysicsEngine;

namespace PhysicsEditor
{
	class InspectorDrawer
	{
		public:
			InspectorDrawer();
			virtual ~InspectorDrawer() = 0;

			virtual void render(Component* component) = 0;
	};
}



#endif