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

	typedef struct Cell
	{
		float tx0;
		float ty0;
		float tz0;
		float tx1;
		float ty1;
		float tz1;
		int nodeIndex;
	}Cell;

	class Octtree
	{
		private:
			int maxNumOfObjectsPerNode;
			int depth;
			Bounds bounds;
			std::vector<Node> nodes;
			std::vector<float> lines; 

		public:
			Octtree();
			~Octtree();

			void clear();
			void create(Bounds bounds, int depth, int maxNumOfObjectsPerNode);
			void insert(Sphere sphere, Guid id);
			Object* intersect(Ray ray);

			int getDepth() const;
			Bounds getBounds() const;
			std::vector<float> getLines() const;

		private:
			int firstNode(float tx0, float ty0, float tz0, float txm, float tym, float tzm);
			int nextNode(float tx, int i, float ty, int j, float tz, int k);
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