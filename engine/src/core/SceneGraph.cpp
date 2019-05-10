#include "../../include/core/SceneGraph.h"

#include <iostream>
#include <stack>

using namespace PhysicsEngine;

SceneGraph::SceneGraph()
{

}

SceneGraph::~SceneGraph()
{

}

void SceneGraph::init(World* world)
{
	int numberOfTransforms = world->getNumberOfComponents<Transform>();

	nodes.resize(numberOfTransforms);

	Transform* root = NULL;
	for(int i = 0; i < world->getNumberOfComponents<Transform>(); i++){
		Transform* transform = world->getComponentByIndex<Transform>(i);

		if(transform->parentId == Guid::INVALID){
			if(root == NULL){
				root = transform;
			}
			else{
				std::cout << "Error: Multiple root indices detected during scene graph initialization" << std::endl;
				return;
			}
		}
	}

	if(root == NULL){
		std::cout << "Error: No root transform found during scene graph initialization" << std::endl;
		return;
	}

	for(int i = 0; i < world->getNumberOfComponents<Transform>(); i++){
		Transform* transform = world->getComponentByIndex<Transform>(i);


	}
}

void SceneGraph::update()
{

}