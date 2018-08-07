#include <iostream>
#include <stack>

#include "../../include/core/OctTree.h"
#include "../../include/core/Log.h"

using namespace PhysicsEngine;

Node::Node()
{
	
}

Node::~Node()
{

}

bool Node::containsAny()
{
	return indices.size() > 0;
}

bool Node::contains(int index)
{
	for (unsigned int i = 0; i < indices.size(); i++){
		if (indices[i] == index){
			return true;
		}
	}

	return false;
}

void Node::add(int index)
{
	for (unsigned int i = 0; i < indices.size(); i++){
		if (indices[i] == index){
			return;
		}
	}

	indices.push_back(index);
}

void Node::clear()
{
	indices.clear();
}


Octtree::Octtree()
{

}

Octtree::Octtree(Bounds bounds, int depth)
{
	allocate(bounds, depth);
}

Octtree::~Octtree()
{

}

void Octtree::allocate(Bounds bounds, int depth)
{
	this->bounds = bounds;

	int size = 1;
	int levelSize = 1;
	int d = 0;
	while (d < depth){
		levelSize *= 8;
		size += levelSize;
		d++;
	}
	
	nodes.resize(size);

	nodes[0].bounds = bounds;

	std::stack<int> stack;
	stack.push(0);
	while (!stack.empty()){
		int index = stack.top();
		stack.pop();

		if(8*index + 8 < nodes.size()){
			glm::vec3 extents = nodes[index].bounds.getExtents();

			for (int i = 1; i <= 8; i++){
				nodes[8 * index + i].bounds.size = extents;
			}

			nodes[8 * index + 1].bounds.centre = nodes[index].bounds.centre + 0.5f * glm::vec3(extents.x, extents.y, extents.z);
			nodes[8 * index + 2].bounds.centre = nodes[index].bounds.centre + 0.5f * glm::vec3(extents.x, extents.y, -extents.z);
			nodes[8 * index + 3].bounds.centre = nodes[index].bounds.centre + 0.5f * glm::vec3(extents.x, -extents.y, extents.z);
			nodes[8 * index + 4].bounds.centre = nodes[index].bounds.centre + 0.5f * glm::vec3(-extents.x, extents.y, extents.z);
			nodes[8 * index + 5].bounds.centre = nodes[index].bounds.centre + 0.5f * glm::vec3(extents.x, -extents.y, -extents.z);
			nodes[8 * index + 6].bounds.centre = nodes[index].bounds.centre + 0.5f * glm::vec3(-extents.x, -extents.y, extents.z);
			nodes[8 * index + 7].bounds.centre = nodes[index].bounds.centre + 0.5f * glm::vec3(-extents.x, extents.y, -extents.z);
			nodes[8 * index + 8].bounds.centre = nodes[index].bounds.centre + 0.5f * glm::vec3(-extents.x, -extents.y, -extents.z);

			for (int i = 8 * index + 1; i <= 8 * index + 8; i++){
				stack.push(i);
			}
		}
	}
}

void Octtree::build(std::vector<Collider*> colliders)
{
	this->colliders = colliders;

	//Log::Info("building octtree with %d colliders", colliders.size());

	for (unsigned int i = 0; i < colliders.size(); i++){

		std::stack<int> stack;
		stack.push(0);
		while (!stack.empty()){
			int index = stack.top();
			stack.pop();
			if (colliders[i]->intersect(nodes[index].bounds)){
				nodes[index].add(i);

				//Log::Info("collider %d intersected with node %d", i, index);

				if(8*index+8 < nodes.size()){
					for (int j = 8 * index + 1; j <= 8 * index + 8; j++){
						stack.push(j);
					}
				}
			}
		}
	}
}

std::vector<float> Octtree::getWireframe()
{
	std::vector<float> vertices;
	for(unsigned int i = 0; i < nodes.size(); i++){
		Node* node = &nodes[i];

		if(node->containsAny()){
			glm::vec3 centre = node->bounds.centre;
			glm::vec3 extents = 0.5f * node->bounds.size;

			//Log::Info("extents: %f %f %f", extents.x, extents.y, extents.z);

			vertices.push_back(centre.x - extents.x);
			vertices.push_back(centre.y - extents.y);
			vertices.push_back(centre.z - extents.z);

			vertices.push_back(centre.x + extents.x);
			vertices.push_back(centre.y - extents.y);
			vertices.push_back(centre.z - extents.z);

			vertices.push_back(centre.x + extents.x);
			vertices.push_back(centre.y - extents.y);
			vertices.push_back(centre.z - extents.z);

			vertices.push_back(centre.x + extents.x);
			vertices.push_back(centre.y + extents.y);
			vertices.push_back(centre.z - extents.z);

			vertices.push_back(centre.x + extents.x);
			vertices.push_back(centre.y + extents.y);
			vertices.push_back(centre.z - extents.z);

			vertices.push_back(centre.x - extents.x);
			vertices.push_back(centre.y + extents.y);
			vertices.push_back(centre.z - extents.z);

			vertices.push_back(centre.x - extents.x);
			vertices.push_back(centre.y + extents.y);
			vertices.push_back(centre.z - extents.z);

			vertices.push_back(centre.x - extents.x);
			vertices.push_back(centre.y - extents.y);
			vertices.push_back(centre.z - extents.z);

			//
			vertices.push_back(centre.x - extents.x);
			vertices.push_back(centre.y - extents.y);
			vertices.push_back(centre.z + extents.z);

			vertices.push_back(centre.x + extents.x);
			vertices.push_back(centre.y - extents.y);
			vertices.push_back(centre.z + extents.z);

			vertices.push_back(centre.x + extents.x);
			vertices.push_back(centre.y - extents.y);
			vertices.push_back(centre.z + extents.z);

			vertices.push_back(centre.x + extents.x);
			vertices.push_back(centre.y + extents.y);
			vertices.push_back(centre.z + extents.z);

			vertices.push_back(centre.x + extents.x);
			vertices.push_back(centre.y + extents.y);
			vertices.push_back(centre.z + extents.z);

			vertices.push_back(centre.x - extents.x);
			vertices.push_back(centre.y + extents.y);
			vertices.push_back(centre.z + extents.z);

			vertices.push_back(centre.x - extents.x);
			vertices.push_back(centre.y + extents.y);
			vertices.push_back(centre.z + extents.z);

			vertices.push_back(centre.x - extents.x);
			vertices.push_back(centre.y - extents.y);
			vertices.push_back(centre.z + extents.z);

			//
			vertices.push_back(centre.x - extents.x);
			vertices.push_back(centre.y - extents.y);
			vertices.push_back(centre.z - extents.z);

			vertices.push_back(centre.x - extents.x);
			vertices.push_back(centre.y - extents.y);
			vertices.push_back(centre.z + extents.z);

			vertices.push_back(centre.x + extents.x);
			vertices.push_back(centre.y - extents.y);
			vertices.push_back(centre.z - extents.z);

			vertices.push_back(centre.x + extents.x);
			vertices.push_back(centre.y - extents.y);
			vertices.push_back(centre.z + extents.z);

			vertices.push_back(centre.x + extents.x);
			vertices.push_back(centre.y + extents.y);
			vertices.push_back(centre.z - extents.z);

			vertices.push_back(centre.x + extents.x);
			vertices.push_back(centre.y + extents.y);
			vertices.push_back(centre.z + extents.z);

			vertices.push_back(centre.x - extents.x);
			vertices.push_back(centre.y + extents.y);
			vertices.push_back(centre.z - extents.z);

			vertices.push_back(centre.x - extents.x);
			vertices.push_back(centre.y + extents.y);
			vertices.push_back(centre.z + extents.z);
		}
	}

	return vertices;
}