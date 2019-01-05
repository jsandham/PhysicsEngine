#include <iostream>
#include <algorithm>
#include <stack>

#include "../../include/core/OctTree.h"
#include "../../include/core/Log.h"
#include "../../include/core/Geometry.h"

using namespace PhysicsEngine;

Octtree::Octtree(Bounds bounds, int depth)
{
	this->bounds = bounds;
	this->depth = depth;

	int d = 0;
	int levelSize = 1;
	int totalSize = levelSize;
	while(d < depth){
		levelSize *= 8;
		d++;
		totalSize += levelSize;
	}

	nodes.resize(totalSize);

	std::cout << "Number of nodes allocated: " << nodes.size() << " Size of a Node is: " << sizeof(nodes[0]) << std::endl;

	nodes[0].extent = 0.5f * bounds.size;
	nodes[0].centre = bounds.centre;

	std::cout << "root node centre: " << nodes[0].centre.x << " " << nodes[0].centre.y << " " << nodes[0].centre.z << " root node extents: " << nodes[0].extent.x << " " << nodes[0].extent.y << " " << nodes[0].extent.z << std::endl;

	std::stack<int> stack;

	stack.push(0);
	while(!stack.empty()){
		int currentIndex = stack.top();
		stack.pop();

		if(8*currentIndex + 8 < nodes.size()){
			for(int i = -1; i <= 1; i += 2){
				for(int j = -1; j <= 1; j += 2){
					for(int k = -1; k <= 1; k += 2){

						glm::vec3 newExtent = 0.5f * nodes[currentIndex].extent;
						glm::vec3 newCentre;
						newCentre.x = nodes[currentIndex].centre.x + i * 0.5f * nodes[currentIndex].extent.x;
						newCentre.y = nodes[currentIndex].centre.y + j * 0.5f * nodes[currentIndex].extent.y;
						newCentre.z = nodes[currentIndex].centre.z + k * 0.5f * nodes[currentIndex].extent.z;

						int quadrant = 0;
						for(int l = 0; l < 3; l++){
							float delta = newCentre[l] - nodes[currentIndex].centre[l];
							if(delta > 0.0f){ quadrant |= (1 << l); }
						}

						int index = 8*currentIndex + quadrant + 1;
						nodes[index].extent = newExtent;
						nodes[index].centre = newCentre;

						stack.push(index);
					}
				}
			}
		}
	}

	lines.resize(6*12*nodes.size());

	for(unsigned int i = 0; i < nodes.size(); i++){
		Node* node = &nodes[i];

		// top
		lines[6*12*i] = node->centre.x - node->extent.x;
		lines[6*12*i + 1] = node->centre.y + node->extent.y;
		lines[6*12*i + 2] = node->centre.z + node->extent.z;
		lines[6*12*i + 3] = node->centre.x + node->extent.x;
		lines[6*12*i + 4] = node->centre.y + node->extent.y;
		lines[6*12*i + 5] = node->centre.z + node->extent.z;

		lines[6*12*i + 6] = node->centre.x + node->extent.x;
		lines[6*12*i + 7] = node->centre.y + node->extent.y;
		lines[6*12*i + 8] = node->centre.z + node->extent.z;
		lines[6*12*i + 9] = node->centre.x + node->extent.x;
		lines[6*12*i + 10] = node->centre.y - node->extent.y;
		lines[6*12*i + 11] = node->centre.z + node->extent.z;

		lines[6*12*i + 12] = node->centre.x + node->extent.x;
		lines[6*12*i + 13] = node->centre.y - node->extent.y;
		lines[6*12*i + 14] = node->centre.z + node->extent.z;
		lines[6*12*i + 15] = node->centre.x - node->extent.x;
		lines[6*12*i + 16] = node->centre.y - node->extent.y;
		lines[6*12*i + 17] = node->centre.z + node->extent.z;

		lines[6*12*i + 18] = node->centre.x - node->extent.x;
		lines[6*12*i + 19] = node->centre.y - node->extent.y;
		lines[6*12*i + 20] = node->centre.z + node->extent.z;
		lines[6*12*i + 21] = node->centre.x - node->extent.x;
		lines[6*12*i + 22] = node->centre.y + node->extent.y;
		lines[6*12*i + 23] = node->centre.z + node->extent.z;

		// bottom
		lines[6*12*i + 24] = node->centre.x - node->extent.x;
		lines[6*12*i + 25] = node->centre.y + node->extent.y;
		lines[6*12*i + 26] = node->centre.z - node->extent.z;
		lines[6*12*i + 27] = node->centre.x + node->extent.x;
		lines[6*12*i + 28] = node->centre.y + node->extent.y;
		lines[6*12*i + 29] = node->centre.z - node->extent.z;

		lines[6*12*i + 30] = node->centre.x + node->extent.x;
		lines[6*12*i + 31] = node->centre.y + node->extent.y;
		lines[6*12*i + 32] = node->centre.z - node->extent.z;
		lines[6*12*i + 33] = node->centre.x + node->extent.x;
		lines[6*12*i + 34] = node->centre.y - node->extent.y;
		lines[6*12*i + 35] = node->centre.z - node->extent.z;

		lines[6*12*i + 36] = node->centre.x + node->extent.x;
		lines[6*12*i + 37] = node->centre.y - node->extent.y;
		lines[6*12*i + 38] = node->centre.z - node->extent.z;
		lines[6*12*i + 39] = node->centre.x - node->extent.x;
		lines[6*12*i + 40] = node->centre.y - node->extent.y;
		lines[6*12*i + 41] = node->centre.z - node->extent.z;

		lines[6*12*i + 42] = node->centre.x - node->extent.x;
		lines[6*12*i + 43] = node->centre.y - node->extent.y;
		lines[6*12*i + 44] = node->centre.z - node->extent.z;
		lines[6*12*i + 45] = node->centre.x - node->extent.x;
		lines[6*12*i + 46] = node->centre.y + node->extent.y;
		lines[6*12*i + 47] = node->centre.z - node->extent.z;

		// sides
		lines[6*12*i + 48] = node->centre.x - node->extent.x;
		lines[6*12*i + 49] = node->centre.y + node->extent.y;
		lines[6*12*i + 50] = node->centre.z + node->extent.z;
		lines[6*12*i + 51] = node->centre.x - node->extent.x;
		lines[6*12*i + 52] = node->centre.y + node->extent.y;
		lines[6*12*i + 53] = node->centre.z - node->extent.z;

		lines[6*12*i + 54] = node->centre.x + node->extent.x;
		lines[6*12*i + 55] = node->centre.y + node->extent.y;
		lines[6*12*i + 56] = node->centre.z + node->extent.z;
		lines[6*12*i + 57] = node->centre.x + node->extent.x;
		lines[6*12*i + 58] = node->centre.y + node->extent.y;
		lines[6*12*i + 59] = node->centre.z - node->extent.z;

		lines[6*12*i + 60] = node->centre.x + node->extent.x;
		lines[6*12*i + 61] = node->centre.y - node->extent.y;
		lines[6*12*i + 62] = node->centre.z + node->extent.z;
		lines[6*12*i + 63] = node->centre.x + node->extent.x;
		lines[6*12*i + 64] = node->centre.y - node->extent.y;
		lines[6*12*i + 65] = node->centre.z - node->extent.z;

		lines[6*12*i + 66] = node->centre.x - node->extent.x;
		lines[6*12*i + 67] = node->centre.y - node->extent.y;
		lines[6*12*i + 68] = node->centre.z + node->extent.z;
		lines[6*12*i + 69] = node->centre.x - node->extent.x;
		lines[6*12*i + 70] = node->centre.y - node->extent.y;
		lines[6*12*i + 71] = node->centre.z - node->extent.z;
	}

	std::cout << "lines count: " << lines.size() << std::endl;

	tempLines.resize(lines.size());
}

Octtree::~Octtree()
{
	
}

void Octtree::clear()
{
	for(unsigned int i = 0; i < nodes.size(); i++){
		nodes[i].objects.clear();
	}
}

void Octtree::insert(Sphere sphere, Guid id)
{
	Object object;
	object.sphere = sphere;
	object.id = id;

	int currentDepth = 0;

	std::stack<int> stack;

	stack.push(0);
	while(!stack.empty()){
		int nodeIndex = stack.top();
		stack.pop();

		// find quadrant that completely contains the object
		bool straddle = false;
		int quadrant = 0;
		for(int i = 0; i < 3; i++){
			float delta = sphere.centre[i] - nodes[nodeIndex].centre[i];
			if(std::abs(delta) <= sphere.radius){
				straddle = true;
				break;
			}

			if(delta > 0.0f){ quadrant |= (1 << i); }
		}

		if(!straddle && currentDepth < depth){
			stack.push(8*nodeIndex + quadrant + 1);
		}
		else{
			// insert object into current node
			nodes[nodeIndex].objects.push_back(object);
		}

		currentDepth++;
	}
}


// Ray octtree intersection as described in the paper
// "An Efficient Parametric Algorithm for Octree Traversal" by Revelles, Urena, & Lastra
Object* Octtree::intersect(Ray ray)
{
	std::cout << "origin.x: " << ray.origin.x << " origin.y: " << ray.origin.y << " origin.z: " << ray.origin.z << " direction.x: " << ray.direction.x << " direction.y: " << ray.direction.y << " ray.direction.z: " << ray.direction.z << std::endl;

	unsigned int a = 0;
	if(ray.direction.x < 0.0f){
		// ray.origin.x = bounds.size.x - ray.origin.x;
		ray.origin.x = 2.0f * bounds.centre.x - ray.origin.x;
		ray.direction.x = -ray.direction.x;
		a |= 4;
	}

	if(ray.direction.y < 0.0f){
		// ray.origin.y = bounds.size.y - ray.origin.y;
		ray.origin.y = 2.0f * bounds.centre.y - ray.origin.y;
		ray.direction.y = -ray.direction.y;
		a |= 2;
	}

	if(ray.direction.z < 0.0f){
		// ray.origin.z = bounds.size.z - ray.origin.z;
		ray.origin.z = 2.0f * bounds.centre.z - ray.origin.z;
		ray.direction.z = -ray.direction.z;
		a |= 1;
	}

	// if(std::max(tx0, std::max(ty0, tz0)) >= std::min(tx1, std::min(ty1, tz1))){
	// 	return NULL;
	// }

	std::cout << "intersect called " << bounds.size.x << " " << bounds.size.y << " " << bounds.size.z << std::endl;

	std::stack<int> stack;
	stack.push(0);
	while(!stack.empty()){
		int nodeIndex = stack.top();
		stack.pop();

		if(8*nodeIndex + 8 >= nodes.size()){
			// nodeIndex is a child node. add objects in this node to list to search
			//std::cout << "node index: " << nodeIndex << " is a child" << std::endl;
			continue;
		}

		float xmin = nodes[nodeIndex].centre.x - nodes[nodeIndex].extent.x;
		float xmax = nodes[nodeIndex].centre.x + nodes[nodeIndex].extent.x;
		float ymin = nodes[nodeIndex].centre.y - nodes[nodeIndex].extent.y;
		float ymax = nodes[nodeIndex].centre.y + nodes[nodeIndex].extent.y;
		float zmin = nodes[nodeIndex].centre.z - nodes[nodeIndex].extent.z;
		float zmax = nodes[nodeIndex].centre.z + nodes[nodeIndex].extent.z;

		float tx0 = (xmin - ray.origin.x) / ray.direction.x;
		float tx1 = (xmax - ray.origin.x) / ray.direction.x;
		float ty0 = (ymin - ray.origin.y) / ray.direction.y;
		float ty1 = (ymax - ray.origin.y) / ray.direction.y;
		float tz0 = (zmin - ray.origin.z) / ray.direction.z;
		float tz1 = (zmax - ray.origin.z) / ray.direction.z;

		std::cout << "node index: " << nodeIndex << " tx0: " << tx0 << " ty0: " << ty0 << " tz0: " << tz0 << " tx1: " << tx1 << " ty1: " << ty1 << " tz1: " << tz1 << std::endl;

		// tx1, ty1, and tz1 cannot be negative if the ray intersects the octtree
		if(tx1 < 0.0f || ty1 < 0.0f || tz1 < 0.0f){
			continue;
		}

		if(std::max(tx0, std::max(ty0, tz0)) >= std::min(tx1, std::min(ty1, tz1))){
			continue;
		}

		float txm = 0.5f * (tx0 + tx1);
		float tym = 0.5f * (ty0 + ty1);
		float tzm = 0.5f * (tz0 + tz1);

		// find first node
		int localNodeIndex = firstNode(tx0, ty0, tz0, txm, tym, tzm);

		while(localNodeIndex < 8){
			stack.push(8*nodeIndex + (localNodeIndex ^ a) + 1);

			switch(localNodeIndex)
			{
				case 0:
					localNodeIndex = nextNode(txm, 4, tym, 2, tzm, 1);
					break;
				case 1:
					localNodeIndex = nextNode(txm, 5, tym, 3, tz1, 8);
					break;
				case 2:
					localNodeIndex = nextNode(txm, 6, ty1, 8, tzm, 3);
					break;
				case 3:
					localNodeIndex = nextNode(txm, 7, ty1, 8, tz1, 8);
					break;
				case 4:
					localNodeIndex = nextNode(tx1, 8, tym, 6, tzm, 5);
					break;
				case 5:
					localNodeIndex = nextNode(tx1, 8, tym, 7, tz1, 8);
					break;
				case 6:
					localNodeIndex = nextNode(tx1, 8, ty1, 8, tzm, 7);
					break;
				case 7:
					localNodeIndex = 8;
					break;
			}
		}
	}

	return NULL;
}

// First node selection as described in the paper
// "An Efficient Parametric Algorithm for Octree Traversal" by Revelles, Urena, & Lastra
int Octtree::firstNode(float tx0, float ty0, float tz0, float txm, float tym, float tzm)
{
	int index = 0;
	if(tx0 >= std::max(ty0, tz0)){ // enters YZ plane
		if(tym < tx0){ index = index | (1 << 1); }
		if(tzm < tx0){ index = index | (1 << 2); }
	}
	else if(ty0 >= std::max(tx0, tz0)){ // enters XZ plane
		if(txm < ty0){ index = index | (1 << 0); }
		if(tzm < ty0){ index = index | (1 << 2); }
	}
	else // enters XY plane
	{
		if(txm < tz0){ index = index | (1 << 0); }
		if(tym < tz0){ index = index | (1 << 1); }
	}

	return index;
}

// Next node selection as described in the paper
// "An Efficient Parametric Algorithm for Octree Traversal" by Revelles, Urena, & Lastra
int Octtree::nextNode(float tx, int i, float ty, int j, float tz, int k)
{
	if(tx < std::min(ty, tz)){  // YZ plane
		return i;
	}
	else if(ty < std::min(tx, tz)){ // XZ plane
		return j;
	}
	else{ // XY plane
		return k;
	}
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

std::vector<float> Octtree::getLinesTemp()
{
	// tempLines.clear();
	for(unsigned int i = 0; i < tempLines.size(); i++){
		tempLines[i] = 0.0f;
	}

	int index = 0;
	for(unsigned int i = 0; i < nodes.size(); i++){
		Node* node = &nodes[i];
		if(node->objects.size() > 0){
			// top
			tempLines[6*12*index] = node->centre.x - node->extent.x;
			tempLines[6*12*index + 1] = node->centre.y + node->extent.y;
			tempLines[6*12*index + 2] = node->centre.z + node->extent.z;
			tempLines[6*12*index + 3] = node->centre.x + node->extent.x;
			tempLines[6*12*index + 4] = node->centre.y + node->extent.y;
			tempLines[6*12*index + 5] = node->centre.z + node->extent.z;

			tempLines[6*12*index + 6] = node->centre.x + node->extent.x;
			tempLines[6*12*index + 7] = node->centre.y + node->extent.y;
			tempLines[6*12*index + 8] = node->centre.z + node->extent.z;
			tempLines[6*12*index + 9] = node->centre.x + node->extent.x;
			tempLines[6*12*index + 10] = node->centre.y - node->extent.y;
			tempLines[6*12*index + 11] = node->centre.z + node->extent.z;

			tempLines[6*12*index + 12] = node->centre.x + node->extent.x;
			tempLines[6*12*index + 13] = node->centre.y - node->extent.y;
			tempLines[6*12*index + 14] = node->centre.z + node->extent.z;
			tempLines[6*12*index + 15] = node->centre.x - node->extent.x;
			tempLines[6*12*index + 16] = node->centre.y - node->extent.y;
			tempLines[6*12*index + 17] = node->centre.z + node->extent.z;

			tempLines[6*12*index + 18] = node->centre.x - node->extent.x;
			tempLines[6*12*index + 19] = node->centre.y - node->extent.y;
			tempLines[6*12*index + 20] = node->centre.z + node->extent.z;
			tempLines[6*12*index + 21] = node->centre.x - node->extent.x;
			tempLines[6*12*index + 22] = node->centre.y + node->extent.y;
			tempLines[6*12*index + 23] = node->centre.z + node->extent.z;

			// bottom
			tempLines[6*12*index + 24] = node->centre.x - node->extent.x;
			tempLines[6*12*index + 25] = node->centre.y + node->extent.y;
			tempLines[6*12*index + 26] = node->centre.z - node->extent.z;
			tempLines[6*12*index + 27] = node->centre.x + node->extent.x;
			tempLines[6*12*index + 28] = node->centre.y + node->extent.y;
			tempLines[6*12*index + 29] = node->centre.z - node->extent.z;

			tempLines[6*12*index + 30] = node->centre.x + node->extent.x;
			tempLines[6*12*index + 31] = node->centre.y + node->extent.y;
			tempLines[6*12*index + 32] = node->centre.z - node->extent.z;
			tempLines[6*12*index + 33] = node->centre.x + node->extent.x;
			tempLines[6*12*index + 34] = node->centre.y - node->extent.y;
			tempLines[6*12*index + 35] = node->centre.z - node->extent.z;

			tempLines[6*12*index + 36] = node->centre.x + node->extent.x;
			tempLines[6*12*index + 37] = node->centre.y - node->extent.y;
			tempLines[6*12*index + 38] = node->centre.z - node->extent.z;
			tempLines[6*12*index + 39] = node->centre.x - node->extent.x;
			tempLines[6*12*index + 40] = node->centre.y - node->extent.y;
			tempLines[6*12*index + 41] = node->centre.z - node->extent.z;

			tempLines[6*12*index + 42] = node->centre.x - node->extent.x;
			tempLines[6*12*index + 43] = node->centre.y - node->extent.y;
			tempLines[6*12*index + 44] = node->centre.z - node->extent.z;
			tempLines[6*12*index + 45] = node->centre.x - node->extent.x;
			tempLines[6*12*index + 46] = node->centre.y + node->extent.y;
			tempLines[6*12*index + 47] = node->centre.z - node->extent.z;

			// sides
			tempLines[6*12*index + 48] = node->centre.x - node->extent.x;
			tempLines[6*12*index + 49] = node->centre.y + node->extent.y;
			tempLines[6*12*index + 50] = node->centre.z + node->extent.z;
			tempLines[6*12*index + 51] = node->centre.x - node->extent.x;
			tempLines[6*12*index + 52] = node->centre.y + node->extent.y;
			tempLines[6*12*index + 53] = node->centre.z - node->extent.z;

			tempLines[6*12*index + 54] = node->centre.x + node->extent.x;
			tempLines[6*12*index + 55] = node->centre.y + node->extent.y;
			tempLines[6*12*index + 56] = node->centre.z + node->extent.z;
			tempLines[6*12*index + 57] = node->centre.x + node->extent.x;
			tempLines[6*12*index + 58] = node->centre.y + node->extent.y;
			tempLines[6*12*index + 59] = node->centre.z - node->extent.z;

			tempLines[6*12*index + 60] = node->centre.x + node->extent.x;
			tempLines[6*12*index + 61] = node->centre.y - node->extent.y;
			tempLines[6*12*index + 62] = node->centre.z + node->extent.z;
			tempLines[6*12*index + 63] = node->centre.x + node->extent.x;
			tempLines[6*12*index + 64] = node->centre.y - node->extent.y;
			tempLines[6*12*index + 65] = node->centre.z - node->extent.z;

			tempLines[6*12*index + 66] = node->centre.x - node->extent.x;
			tempLines[6*12*index + 67] = node->centre.y - node->extent.y;
			tempLines[6*12*index + 68] = node->centre.z + node->extent.z;
			tempLines[6*12*index + 69] = node->centre.x - node->extent.x;
			tempLines[6*12*index + 70] = node->centre.y - node->extent.y;
			tempLines[6*12*index + 71] = node->centre.z - node->extent.z;

			index++;
		}
	}

	return tempLines;
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