#ifndef __OCTTREE_H__
#define __OCTTREE_H__

#include <vector>

#include "Guid.h"
#include "Sphere.h"
#include "Bounds.h"
#include "Capsule.h"
#include "Ray.h"

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"
#include "../glm/gtc/matrix_transform.hpp"
#include "../glm/gtc/type_ptr.hpp"

#include "../components/Collider.h"


namespace PhysicsEngine
{
	typedef struct Object
	{
		Guid id;
		Sphere sphere;  //Primitive which sphere, bounds, capsule, triangle derive from?
	}Object;

	typedef struct Node 
	{
		glm::vec3 centre;
		glm::vec3 extent;
		std::vector<Object> objects;
	}Node;

	class Octtree
	{
		private:
			int depth;
			Bounds bounds;
			std::vector<Node> nodes;

			std::vector<Object> tempObjects; ///remove later. Just here for testing

		public:
			Octtree(Bounds bounds, int depth);
			~Octtree();

			void insert(Sphere sphere, Guid id);

			Object* intersect(Ray ray);





			void tempClear();
			void tempInsert(Sphere sphere, Guid id);
			Object* tempIntersect(Ray ray);
	};














	// class Node
	// {
	// 	private:
	// 		std::vector<int> indices;

	// 	public:
	// 		Bounds bounds;
			
	// 	public:
	// 		Node();
	// 		~Node();

	// 		bool containsAny();
	// 		bool contains(int index);
	// 		void add(int index);
	// 		void clear();
	// };


	// class Octtree
	// {
	// 	private:
	// 		Bounds bounds;

	// 		std::vector<Node> nodes;
	// 		std::vector<Collider*> colliders;

	// 	public:
	// 		Octtree();
	// 		Octtree(Bounds bounds, int depth);
	// 		~Octtree();

	// 		void allocate(Bounds bounds, int depth);
	// 		void build(std::vector<Collider*> colliders);

	// 		std::vector<float> getWireframe();
	// };
}

#endif