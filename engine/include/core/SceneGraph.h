#ifndef __SCENEGRAPH_H__
#define __SCENEGRAPH_H__

#include <vector>

#include "../glm/glm.hpp"

#include "World.h"

namespace PhysicsEngine
{
	struct SceneNode
	{
		int parent;
		std::vector<int> children;
		glm::mat4 localTransform;
		glm::mat4 worldTransform;
	};

	class SceneGraph
	{
		private:
			std::vector<SceneNode> nodes;
			std::vector<int> dirtyIndices;

		public:
			SceneGraph();
			~SceneGraph();

			void init(World* world);
			void update();
	};	
}

#endif