#ifndef __CAMERA_DRAWER_H__
#define __CAMERA_DRAWER_H__

#include "InspectorDrawer.h"

namespace PhysicsEditor
{
	class CameraDrawer : public InspectorDrawer
	{
		public:
			CameraDrawer();
			~CameraDrawer();

			void render(World* world, EditorClipboard& clipboard, Guid id);
	};
}

#endif