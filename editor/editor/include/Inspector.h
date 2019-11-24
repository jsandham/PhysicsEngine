#ifndef __INSPECTOR_H__
#define __INSPECTOR_H__

#include <vector>

#include "InspectorDrawer.h"
#include "EditorScene.h"

#include "core/World.h"
#include "core/Entity.h"

using namespace PhysicsEngine;

namespace PhysicsEditor
{
	class Inspector
	{
		private:
			std::vector<InspectorDrawer> drawers;

		public:
			Inspector();
			~Inspector();

			void render(World* world, Entity* entity, EditorScene& scene, bool isOpenedThisFrame);




			// move to imgui extensions?
			bool BeginAddComponentDropdown(std::string filter, std::string& componentToAdd);
			void EndAddComponentDropdown();
	};
}

#endif