#ifndef __TRANSFORM_DRAWER_H__
#define __TRANSFORM_DRAWER_H__

#include "InspectorDrawer.h"

namespace PhysicsEditor
{
	class TransformDrawer : public InspectorDrawer
	{
		public:
			TransformDrawer();
			~TransformDrawer();

			void render(World* world, EditorUI& ui, Guid entityId, Guid componentId);
	};
}

#endif