#ifndef __HIERARCHY_H__
#define __HIERARCHY_H__

#include <vector>

#include "core/Entity.h"
#include "core/World.h"

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

			void render(World world, bool isOpenedThisFrame);

			//bool begin

			Entity* getSelectedEntity();
	};
}

#endif