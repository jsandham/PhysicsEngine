#include <iostream>
#include <cstddef>
#include <ctime>

#include "../../include/systems/RenderSystem.h"

#include "../../include/graphics/ForwardRenderer.h"
#include "../../include/graphics/GraphicsQuery.h"

#include "../../include/core/Input.h"
#include "../../include/core/Time.h"

using namespace PhysicsEngine;

RenderSystem::RenderSystem()
{
	type = 0;
}

RenderSystem::RenderSystem(std::vector<char> data)
{
	size_t index = sizeof(char);
	type = *reinterpret_cast<int*>(&data[index]);
	index += sizeof(int);
	order = *reinterpret_cast<int*>(&data[index]);

	if(type != 0){
		std::cout << "Error: System type (" << type << ") found in data array is invalid" << std::endl;
	}
}

RenderSystem::~RenderSystem()
{
}

void RenderSystem::init(World* world)
{
	this->world = world;

	forwardRenderer.init(world);
	//deferredRenderer.init(world);
	debugRenderer.init(world);
}

void RenderSystem::update(Input input)
{
	// if(getKeyDown(input, KeyCode::N)){
	// 	std::cout << "N pressed " << world->getNumberOfEntities() << std::endl;
	// 	Entity* entity = world->createEntity();
	// 	if(entity != NULL){
	// 		Transform* transform = entity->addComponent<Transform>(world);
	// 		transform->position = glm::vec3(0.0f, 0.0f, 0.0f);
	// 		transform->rotation = glm::quat(0.0f, 0.0f, 0.0f, 1.0f);
	// 		transform->scale = glm::vec3(1.0f, 1.0f, 1.0f);

	// 		Guid materialId("a8b069dc-1875-434d-8b7d-2db63ec6e21c");
	// 		Guid meshId("147e88cd-5d29-4cac-9bd0-e2c40684fa0a"); 

	// 		MeshRenderer* meshRenderer = entity->addComponent<MeshRenderer>(world);
	// 		meshRenderer->isStatic = false;
	// 		meshRenderer->materialId = materialId;
	// 		meshRenderer->meshId = meshId;


	// 		std::cout << "newly created entity id: " << entity->entityId.toString() << " transform id: " << transform->componentId.toString() << " mesh renderer id: " << meshRenderer->componentId.toString() << std::endl;
	// 	}

	// 	std::vector<std::pair<Guid,int> > temp = world->getComponentIdsMarkedCreated();

	// 	std::cout << "number of component ids marked created: " << temp.size() << std::endl;
	// }



	//deferredRenderer.update(input);
	forwardRenderer.update(input);

	if(world->debug){
		GraphicsQuery query = forwardRenderer.getGraphicsQuery();
		GraphicsDebug debug = forwardRenderer.getGraphicsDebug();

		debugRenderer.update(input, debug, query);
	}
}