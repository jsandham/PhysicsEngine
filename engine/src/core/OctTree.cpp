#include <iostream>
#include <stack>

#include "../../include/core/OctTree.h"
#include "../../include/core/Log.h"
#include "../../include/core/Geometry.h"

using namespace PhysicsEngine;

Octtree::Octtree(Bounds bounds, int depth)
{
	this->bounds = bounds;
	this->depth = depth;

	int size = 1;
	int d = 0;
	while(d < depth){
		size *= 8;
		d++;
	}

	nodes.resize(size + 1);

	std::cout << "Number of nodes allocated: " << nodes.size() << " Size of a Node is: " << sizeof(nodes[0]) << std::endl;

	nodes[0].extent = bounds.size;
	nodes[0].centre = bounds.centre;

	std::cout << "root node centre: " << nodes[0].centre.x << " " << nodes[0].centre.y << " " << nodes[0].centre.z << " root node size: " << nodes[0].extent.x << " " << nodes[0].extent.y << " " << nodes[0].extent.z << std::endl;

	std::stack<int> stack;

	stack.push(0);
	while(!stack.empty()){
		int currentIndex = stack.top();
		stack.pop();

		if(8*currentIndex + 8 < nodes.size()){
			int quadrant = 1;
			for(int i = -1; i <= 1; i += 2){
				for(int j = -1; j <= 1; j += 2){
					for(int k = -1; k <= 1; k += 2){
						int index = 8*currentIndex + quadrant;
						nodes[index].extent = 0.5f * nodes[currentIndex].extent;
						nodes[index].centre.x = nodes[currentIndex].centre.x + i * 0.25f * nodes[currentIndex].extent.x;
						nodes[index].centre.y = nodes[currentIndex].centre.y + j * 0.25f * nodes[currentIndex].extent.y;
						nodes[index].centre.z = nodes[currentIndex].centre.z + k * 0.25f * nodes[currentIndex].extent.z;

						stack.push(index);

						quadrant++;
					}
				}
			}
		}
	}

	lines.resize(6*12*nodes.size());

	for(unsigned int i = 0; i < nodes.size(); i++){
		Node* node = &nodes[i];

		// top
		lines[6*12*i] = node->centre.x - 0.5f*node->extent.x;
		lines[6*12*i + 1] = node->centre.y + 0.5f*node->extent.y;
		lines[6*12*i + 2] = node->centre.z + 0.5f*node->extent.z;
		lines[6*12*i + 3] = node->centre.x + 0.5f*node->extent.x;
		lines[6*12*i + 4] = node->centre.y + 0.5f*node->extent.y;
		lines[6*12*i + 5] = node->centre.z + 0.5f*node->extent.z;

		lines[6*12*i + 6] = node->centre.x + 0.5f*node->extent.x;
		lines[6*12*i + 7] = node->centre.y + 0.5f*node->extent.y;
		lines[6*12*i + 8] = node->centre.z + 0.5f*node->extent.z;
		lines[6*12*i + 9] = node->centre.x + 0.5f*node->extent.x;
		lines[6*12*i + 10] = node->centre.y - 0.5f*node->extent.y;
		lines[6*12*i + 11] = node->centre.z + 0.5f*node->extent.z;

		lines[6*12*i + 12] = node->centre.x + 0.5f*node->extent.x;
		lines[6*12*i + 13] = node->centre.y - 0.5f*node->extent.y;
		lines[6*12*i + 14] = node->centre.z + 0.5f*node->extent.z;
		lines[6*12*i + 15] = node->centre.x - 0.5f*node->extent.x;
		lines[6*12*i + 16] = node->centre.y - 0.5f*node->extent.y;
		lines[6*12*i + 17] = node->centre.z + 0.5f*node->extent.z;

		lines[6*12*i + 18] = node->centre.x - 0.5f*node->extent.x;
		lines[6*12*i + 19] = node->centre.y - 0.5f*node->extent.y;
		lines[6*12*i + 20] = node->centre.z + 0.5f*node->extent.z;
		lines[6*12*i + 21] = node->centre.x - 0.5f*node->extent.x;
		lines[6*12*i + 22] = node->centre.y + 0.5f*node->extent.y;
		lines[6*12*i + 23] = node->centre.z + 0.5f*node->extent.z;

		// bottom
		lines[6*12*i + 24] = node->centre.x - 0.5f*node->extent.x;
		lines[6*12*i + 25] = node->centre.y + 0.5f*node->extent.y;
		lines[6*12*i + 26] = node->centre.z - 0.5f*node->extent.z;
		lines[6*12*i + 27] = node->centre.x + 0.5f*node->extent.x;
		lines[6*12*i + 28] = node->centre.y + 0.5f*node->extent.y;
		lines[6*12*i + 29] = node->centre.z - 0.5f*node->extent.z;

		lines[6*12*i + 30] = node->centre.x + 0.5f*node->extent.x;
		lines[6*12*i + 31] = node->centre.y + 0.5f*node->extent.y;
		lines[6*12*i + 32] = node->centre.z - 0.5f*node->extent.z;
		lines[6*12*i + 33] = node->centre.x + 0.5f*node->extent.x;
		lines[6*12*i + 34] = node->centre.y - 0.5f*node->extent.y;
		lines[6*12*i + 35] = node->centre.z - 0.5f*node->extent.z;

		lines[6*12*i + 36] = node->centre.x + 0.5f*node->extent.x;
		lines[6*12*i + 37] = node->centre.y - 0.5f*node->extent.y;
		lines[6*12*i + 38] = node->centre.z - 0.5f*node->extent.z;
		lines[6*12*i + 39] = node->centre.x - 0.5f*node->extent.x;
		lines[6*12*i + 40] = node->centre.y - 0.5f*node->extent.y;
		lines[6*12*i + 41] = node->centre.z - 0.5f*node->extent.z;

		lines[6*12*i + 42] = node->centre.x - 0.5f*node->extent.x;
		lines[6*12*i + 43] = node->centre.y - 0.5f*node->extent.y;
		lines[6*12*i + 44] = node->centre.z - 0.5f*node->extent.z;
		lines[6*12*i + 45] = node->centre.x - 0.5f*node->extent.x;
		lines[6*12*i + 46] = node->centre.y + 0.5f*node->extent.y;
		lines[6*12*i + 47] = node->centre.z - 0.5f*node->extent.z;

		// sides
		lines[6*12*i + 48] = node->centre.x - 0.5f*node->extent.x;
		lines[6*12*i + 49] = node->centre.y + 0.5f*node->extent.y;
		lines[6*12*i + 50] = node->centre.z + 0.5f*node->extent.z;
		lines[6*12*i + 51] = node->centre.x - 0.5f*node->extent.x;
		lines[6*12*i + 52] = node->centre.y + 0.5f*node->extent.y;
		lines[6*12*i + 53] = node->centre.z - 0.5f*node->extent.z;

		lines[6*12*i + 54] = node->centre.x + 0.5f*node->extent.x;
		lines[6*12*i + 55] = node->centre.y + 0.5f*node->extent.y;
		lines[6*12*i + 56] = node->centre.z + 0.5f*node->extent.z;
		lines[6*12*i + 57] = node->centre.x + 0.5f*node->extent.x;
		lines[6*12*i + 58] = node->centre.y + 0.5f*node->extent.y;
		lines[6*12*i + 59] = node->centre.z - 0.5f*node->extent.z;

		lines[6*12*i + 60] = node->centre.x + 0.5f*node->extent.x;
		lines[6*12*i + 61] = node->centre.y - 0.5f*node->extent.y;
		lines[6*12*i + 62] = node->centre.z + 0.5f*node->extent.z;
		lines[6*12*i + 63] = node->centre.x + 0.5f*node->extent.x;
		lines[6*12*i + 64] = node->centre.y - 0.5f*node->extent.y;
		lines[6*12*i + 65] = node->centre.z - 0.5f*node->extent.z;

		lines[6*12*i + 66] = node->centre.x - 0.5f*node->extent.x;
		lines[6*12*i + 67] = node->centre.y - 0.5f*node->extent.y;
		lines[6*12*i + 68] = node->centre.z + 0.5f*node->extent.z;
		lines[6*12*i + 69] = node->centre.x - 0.5f*node->extent.x;
		lines[6*12*i + 70] = node->centre.y - 0.5f*node->extent.y;
		lines[6*12*i + 71] = node->centre.z - 0.5f*node->extent.z;
	}

	std::cout << "lines count: " << lines.size() << std::endl;

	std::cout << "octtree contructor finished" << std::endl;
}

Octtree::~Octtree()
{
	
}

void Octtree::insert(Sphere sphere, Guid id)
{
	Object object;
	object.sphere = sphere;
	object.id = id;

	int currentDepth = 0;

	std::stack<Node*> stack;

	stack.push(&nodes[0]);
	while(!stack.empty()){
		Node* current = stack.top();
		stack.pop();

		// find quadrant that completely contains the object
		bool straddle = false;
		int index = 0;
		for(int i = 0; i < 3; i++){

		}


		if(!straddle && currentDepth < depth){
			Node* child = &nodes[index];
			stack.push(child);
		}
		else{
			// insert object into current node
			current->objects.push_back(object);
		}

		currentDepth++;
	}
}

Object* Octtree::intersect(Ray ray)
{
	return NULL;
}

std::vector<float> Octtree::getLines()
{
	return lines;
}











void Octtree::tempClear()
{
	tempObjects.clear();
}

void Octtree::tempInsert(Sphere sphere, Guid id)
{
	Object obj;
	obj.sphere = sphere;
	obj.id = id;

	tempObjects.push_back(obj);
}

Object* Octtree::tempIntersect(Ray ray)
{
	for(int i = 0; i < tempObjects.size(); i++){
		if(Geometry::intersect(ray, tempObjects[i].sphere)){
			return &tempObjects[i];
		}
	}

	return NULL;
}
















































// Node::Node()
// {
	
// }

// Node::~Node()
// {

// }

// bool Node::containsAny()
// {
// 	return indices.size() > 0;
// }

// bool Node::contains(int index)
// {
// 	for (unsigned int i = 0; i < indices.size(); i++){
// 		if (indices[i] == index){
// 			return true;
// 		}
// 	}

// 	return false;
// }

// void Node::add(int index)
// {
// 	for (unsigned int i = 0; i < indices.size(); i++){
// 		if (indices[i] == index){
// 			return;
// 		}
// 	}

// 	indices.push_back(index);
// }

// void Node::clear()
// {
// 	indices.clear();
// }


// Octtree::Octtree()
// {

// }

// Octtree::Octtree(Bounds bounds, int depth)
// {
// 	allocate(bounds, depth);
// }

// Octtree::~Octtree()
// {

// }

// void Octtree::allocate(Bounds bounds, int depth)
// {
// 	this->bounds = bounds;

// 	int size = 1;
// 	int levelSize = 1;
// 	int d = 0;
// 	while (d < depth){
// 		levelSize *= 8;
// 		size += levelSize;
// 		d++;
// 	}
	
// 	nodes.resize(size);

// 	nodes[0].bounds = bounds;

// 	std::stack<int> stack;
// 	stack.push(0);
// 	while (!stack.empty()){
// 		int index = stack.top();
// 		stack.pop();

// 		if(8*index + 8 < nodes.size()){
// 			glm::vec3 extents = nodes[index].bounds.getExtents();

// 			for (int i = 1; i <= 8; i++){
// 				nodes[8 * index + i].bounds.size = extents;
// 			}

// 			nodes[8 * index + 1].bounds.centre = nodes[index].bounds.centre + 0.5f * glm::vec3(extents.x, extents.y, extents.z);
// 			nodes[8 * index + 2].bounds.centre = nodes[index].bounds.centre + 0.5f * glm::vec3(extents.x, extents.y, -extents.z);
// 			nodes[8 * index + 3].bounds.centre = nodes[index].bounds.centre + 0.5f * glm::vec3(extents.x, -extents.y, extents.z);
// 			nodes[8 * index + 4].bounds.centre = nodes[index].bounds.centre + 0.5f * glm::vec3(-extents.x, extents.y, extents.z);
// 			nodes[8 * index + 5].bounds.centre = nodes[index].bounds.centre + 0.5f * glm::vec3(extents.x, -extents.y, -extents.z);
// 			nodes[8 * index + 6].bounds.centre = nodes[index].bounds.centre + 0.5f * glm::vec3(-extents.x, -extents.y, extents.z);
// 			nodes[8 * index + 7].bounds.centre = nodes[index].bounds.centre + 0.5f * glm::vec3(-extents.x, extents.y, -extents.z);
// 			nodes[8 * index + 8].bounds.centre = nodes[index].bounds.centre + 0.5f * glm::vec3(-extents.x, -extents.y, -extents.z);

// 			for (int i = 8 * index + 1; i <= 8 * index + 8; i++){
// 				stack.push(i);
// 			}
// 		}
// 	}
// }

// void Octtree::build(std::vector<Collider*> colliders)
// {
// 	this->colliders = colliders;

// 	//Log::Info("building octtree with %d colliders", colliders.size());

// 	for (unsigned int i = 0; i < colliders.size(); i++){

// 		std::stack<int> stack;
// 		stack.push(0);
// 		while (!stack.empty()){
// 			int index = stack.top();
// 			stack.pop();
// 			if (colliders[i]->intersect(nodes[index].bounds)){
// 				nodes[index].add(i);

// 				//Log::Info("collider %d intersected with node %d", i, index);

// 				if(8*index+8 < nodes.size()){
// 					for (int j = 8 * index + 1; j <= 8 * index + 8; j++){
// 						stack.push(j);
// 					}
// 				}
// 			}
// 		}
// 	}
// }

// std::vector<float> Octtree::getWireframe()
// {
// 	std::vector<float> vertices;
// 	for(unsigned int i = 0; i < nodes.size(); i++){
// 		Node* node = &nodes[i];

// 		if(node->containsAny()){
// 			glm::vec3 centre = node->bounds.centre;
// 			glm::vec3 extents = 0.5f * node->bounds.size;

// 			//Log::Info("extents: %f %f %f", extents.x, extents.y, extents.z);

// 			vertices.push_back(centre.x - extents.x);
// 			vertices.push_back(centre.y - extents.y);
// 			vertices.push_back(centre.z - extents.z);

// 			vertices.push_back(centre.x + extents.x);
// 			vertices.push_back(centre.y - extents.y);
// 			vertices.push_back(centre.z - extents.z);

// 			vertices.push_back(centre.x + extents.x);
// 			vertices.push_back(centre.y - extents.y);
// 			vertices.push_back(centre.z - extents.z);

// 			vertices.push_back(centre.x + extents.x);
// 			vertices.push_back(centre.y + extents.y);
// 			vertices.push_back(centre.z - extents.z);

// 			vertices.push_back(centre.x + extents.x);
// 			vertices.push_back(centre.y + extents.y);
// 			vertices.push_back(centre.z - extents.z);

// 			vertices.push_back(centre.x - extents.x);
// 			vertices.push_back(centre.y + extents.y);
// 			vertices.push_back(centre.z - extents.z);

// 			vertices.push_back(centre.x - extents.x);
// 			vertices.push_back(centre.y + extents.y);
// 			vertices.push_back(centre.z - extents.z);

// 			vertices.push_back(centre.x - extents.x);
// 			vertices.push_back(centre.y - extents.y);
// 			vertices.push_back(centre.z - extents.z);

// 			//
// 			vertices.push_back(centre.x - extents.x);
// 			vertices.push_back(centre.y - extents.y);
// 			vertices.push_back(centre.z + extents.z);

// 			vertices.push_back(centre.x + extents.x);
// 			vertices.push_back(centre.y - extents.y);
// 			vertices.push_back(centre.z + extents.z);

// 			vertices.push_back(centre.x + extents.x);
// 			vertices.push_back(centre.y - extents.y);
// 			vertices.push_back(centre.z + extents.z);

// 			vertices.push_back(centre.x + extents.x);
// 			vertices.push_back(centre.y + extents.y);
// 			vertices.push_back(centre.z + extents.z);

// 			vertices.push_back(centre.x + extents.x);
// 			vertices.push_back(centre.y + extents.y);
// 			vertices.push_back(centre.z + extents.z);

// 			vertices.push_back(centre.x - extents.x);
// 			vertices.push_back(centre.y + extents.y);
// 			vertices.push_back(centre.z + extents.z);

// 			vertices.push_back(centre.x - extents.x);
// 			vertices.push_back(centre.y + extents.y);
// 			vertices.push_back(centre.z + extents.z);

// 			vertices.push_back(centre.x - extents.x);
// 			vertices.push_back(centre.y - extents.y);
// 			vertices.push_back(centre.z + extents.z);

// 			//
// 			vertices.push_back(centre.x - extents.x);
// 			vertices.push_back(centre.y - extents.y);
// 			vertices.push_back(centre.z - extents.z);

// 			vertices.push_back(centre.x - extents.x);
// 			vertices.push_back(centre.y - extents.y);
// 			vertices.push_back(centre.z + extents.z);

// 			vertices.push_back(centre.x + extents.x);
// 			vertices.push_back(centre.y - extents.y);
// 			vertices.push_back(centre.z - extents.z);

// 			vertices.push_back(centre.x + extents.x);
// 			vertices.push_back(centre.y - extents.y);
// 			vertices.push_back(centre.z + extents.z);

// 			vertices.push_back(centre.x + extents.x);
// 			vertices.push_back(centre.y + extents.y);
// 			vertices.push_back(centre.z - extents.z);

// 			vertices.push_back(centre.x + extents.x);
// 			vertices.push_back(centre.y + extents.y);
// 			vertices.push_back(centre.z + extents.z);

// 			vertices.push_back(centre.x - extents.x);
// 			vertices.push_back(centre.y + extents.y);
// 			vertices.push_back(centre.z - extents.z);

// 			vertices.push_back(centre.x - extents.x);
// 			vertices.push_back(centre.y + extents.y);
// 			vertices.push_back(centre.z + extents.z);
// 		}
// 	}

// 	return vertices;
// }