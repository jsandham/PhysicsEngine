#ifndef __OCTTREE_H__
#define __OCTTREE_H__

#include <vector>

#include "Guid.h"
#include "Sphere.h"
#include "Bounds.h"
#include "Capsule.h"
#include "Ray.h"
#include "Line.h"

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
			std::vector<float> lines; 

			std::vector<Object> tempObjects; ///remove later. Just here for testing
			std::vector<float> tempLines;  //remove later. Just here for testing
			static size_t test;

		public:
			Octtree(Bounds bounds, int depth);
			~Octtree();

			void clear();
			void insert(Sphere sphere, Guid id);

			Object* intersect(Ray ray);
			int firstNode(float tx0, float ty0, float tz0, float txm, float tym, float tzm);
			int nextNode(float tx, int i, float ty, int j, float tz, int k);

			std::vector<float> getLines();





			void tempClear();
			void tempInsert(Sphere sphere, Guid id);
			Object* tempIntersect(Ray ray);
			std::vector<float> getLinesTemp();
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