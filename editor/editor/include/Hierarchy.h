#ifndef __HIERARCHY_H__
#define __HIERARCHY_H__

#include <vector>
#include <string>

#include "core/Entity.h"
#include "core/World.h"

#include "../include/EditorScene.h"
#include "../include/EditorClipboard.h"

using namespace PhysicsEngine;

namespace PhysicsEditor
{
	class Hierarchy
	{
		private:
			std::vector<Entity> entities;

		public:
			Hierarchy();
			~Hierarchy();

			void render(World* world, EditorScene& scene, EditorClipboard& clipboard, bool isOpenedThisFrame);
	};
}

#endif