#ifndef __INSPECTOR_DRAWER_H__
#define __INSPECTOR_DRAWER_H__

#include "core/World.h"
#include "core/Guid.h"

#include "EditorClipboard.h"

using namespace PhysicsEngine;

namespace PhysicsEditor
{
	class InspectorDrawer
	{
		public:
			InspectorDrawer();
			virtual ~InspectorDrawer() = 0;

			virtual void render(World* world, EditorClipboard& clipboard, Guid id) = 0;
	};
}



#endif