#ifndef __OCTTREE_H__
#define __OCTTREE_H__

#include <vector>

#include "../components/Collider.h"
#include "Bounds.h"

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"
#include "../glm/gtc/matrix_transform.hpp"
#include "../glm/gtc/type_ptr.hpp"


namespace PhysicsEngine
{
	class Node
	{
		private:
			std::vector<int> indices;

		public:
			Bounds bounds;
			
		public:
			Node();
			~Node();

			bool containsAny();
			bool contains(int index);
			void add(int index);
			void clear();
	};


	class Octtree
	{
		private:
			Bounds bounds;

			std::vector<Node> nodes;
			std::vector<Collider*> colliders;

		public:
			Octtree();
			Octtree(Bounds bounds, int depth);
			~Octtree();

			void allocate(Bounds bounds, int depth);
			void build(std::vector<Collider*> colliders);

			std::vector<float> getWireframe();
	};
}

#endif