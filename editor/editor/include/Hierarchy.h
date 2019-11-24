#ifndef __HIERARCHY_H__
#define __HIERARCHY_H__

#include <vector>
#include <string>

#include "core/Entity.h"
#include "core/World.h"

#include "../include/EditorScene.h"

using namespace PhysicsEngine;

namespace PhysicsEditor
{
	class Hierarchy
	{
		private:
			Entity* selectedEntity;
			std::vector<Entity> entities;

		public:
			Hierarchy();
			~Hierarchy();

			void render(World* world, EditorScene& scene, bool isOpenedThisFrame);

			Entity* getSelectedEntity();
	};
}

#endif