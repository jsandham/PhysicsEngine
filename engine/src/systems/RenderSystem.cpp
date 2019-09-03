#include <iostream>
#include <cstddef>
#include <ctime>

#include "../../include/systems/RenderSystem.h"

#include "../../include/graphics/ForwardRenderer.h"
#include "../../include/graphics/GraphicsQuery.h"

#include "../../include/core/Input.h"
#include "../../include/core/Time.h"
#include "../../include/core/Log.h"

using namespace PhysicsEngine;

RenderSystem::RenderSystem()
{

}

RenderSystem::RenderSystem(std::vector<char> data)
{
	deserialize(data);
}

RenderSystem::~RenderSystem()
{
}

std::vector<char> RenderSystem::serialize()
{
	size_t numberOfBytes = sizeof(int);
	std::vector<char> data(numberOfBytes);

	memcpy(&data[0], &order, sizeof(int));

	return data;
}

void RenderSystem::deserialize(std::vector<char> data)
{
	order = *reinterpret_cast<int*>(&data[0]);
}

void RenderSystem::init(World* world)
{
	this->world = world;

	forwardRenderer.init(world);
	//deferredRenderer.init(world);
	debugRenderer.init(world);

	testId = Guid::INVALID;
}

void RenderSystem::update(Input input)
{
	Log::info("Render update called\n");

	// if(getKeyDown(input, KeyCode::N)){
	// 	std::cout << "N pressed " << world->getNumberOfEntities() << std::endl;
	// 	Entity* entity = world->createEntity();
	// 	testId = entity->entityId; 
	// 	if(entity != NULL){
	// 		Transform* transform = entity->addComponent<Transform>(world);
	// 		transform->position = glm::vec3(0.0f, 0.0f, 0.0f);
	// 		transform->rotation = glm::quat(0.0f, 0.0f, 0.0f, 1.0f);
	// 		transform->scale = glm::vec3(1.0f, 1.0f, 1.0f);

	// 		Guid materialId("a8b069dc-1875-434d-8b7d-2db63ec6e21c");
	// 		Guid meshId("147e88cd-5d29-4cac-9bd0-e2c40684fa0a"); 

	// 		MeshRenderer* meshRenderer = entity->addComponent<MeshRenderer>(world);
	// 		meshRenderer->isStatic = false;
	// 		meshRenderer->materialIds[0] = materialId;
	// 		meshRenderer->meshId = meshId;

	// 		//Rigidbody* rigidbody = entity->addComponent<Rigidbody>(world);


	// 		// std::cout << "newly created entity id: " << entity->entityId.toString() << " transform id: " << transform->componentId.toString() << " mesh renderer id: " << meshRenderer->componentId.toString() << " and rigidbody id: " << rigidbody->componentId.toString() << std::endl;
	// 		std::cout << "newly created entity id: " << entity->entityId.toString() << " transform id: " << transform->componentId.toString() << " mesh renderer id: " << meshRenderer->componentId.toString() << std::endl;
	// 	}

	// 	std::vector<triple<Guid, Guid, int> > temp = world->getComponentIdsMarkedCreated();

	// 	std::cout << "number of component ids marked created: " << temp.size() << std::endl;
	// }

	// if(getKeyDown(input, KeyCode::M) && testId != Guid::INVALID){
	// 	std::cout << "M pressed " << world->getNumberOfEntities() << std::endl;
	// 	Entity* entity = world->getEntity(testId);
	// 	if(entity != NULL){
	// 		entity->latentDestroy(world);
	// 		testId = Guid::INVALID;
	// 	}
	// }

	if(getKeyDown(input, KeyCode::V)){
		std::cout << "V pressed" << std::endl;
		//std::cout << "transform instance type: " << Component::getInstanceType<Transform>() << " mesh renderer: " << Component::getInstanceType<MeshRenderer>() << " sphere collider: " << Component::getInstanceType<SphereCollider>() << std::endl;
		int index = world->getNumberOfEntities();
		//std::cout << "Total number of entities: " << index << std::endl;
		if(index > 0){
			Entity* entity = world->getEntityByIndex(index - 1);
			std::cout << "Calling latent Destroy on entity id: " << entity->entityId.toString() << " at global index: " << index - 1 << std::endl;
			entity->latentDestroy(world);
		}
	}

	//deferredRenderer.update(input);
	forwardRenderer.update(input);

	if(world->debug){
		GraphicsQuery query = forwardRenderer.getGraphicsQuery();
		GraphicsDebug debug = forwardRenderer.getGraphicsDebug();

		debugRenderer.update(input, debug, query);
	}
}