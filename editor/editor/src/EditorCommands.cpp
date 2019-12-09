#include "../include/EditorCommands.h"

#include "core/Entity.h"

using namespace PhysicsEditor;
using namespace PhysicsEngine;

// Hierarchy commands

CreateEntityCommand::CreateEntityCommand(World* world)
{
	this->world = world;
}

void CreateEntityCommand::execute()
{
	if (entityData.size() == 0) {
		Entity* entity = world->createEntity();
		Transform* transform = entity->addComponent<Transform>(world);
		entityData = entity->serialize();
		transformData = transform->serialize();
	}
	else {
		Entity* entity = world->createEntity(entityData);
		Transform* transform = entity->addComponent<Transform>(world, transformData);
	}
}

void CreateEntityCommand::undo()
{
	Entity temp(entityData);

	world->latentDestroyEntity(temp.entityId);
}


CreateCameraCommand::CreateCameraCommand(World* world)
{
	this->world = world;
}

void CreateCameraCommand::execute()
{
	if (entityData.size() == 0) {
		Entity* entity = world->createEntity();
		Transform* transform = entity->addComponent<Transform>(world);
		Camera* camera = entity->addComponent<Camera>(world);
		entityData = entity->serialize();
		transformData = transform->serialize();
		cameraData = camera->serialize();
	}
	else{
		Entity* entity = world->createEntity(entityData);
		Transform* transform = entity->addComponent<Transform>(world, transformData);
		Camera* camera = entity->addComponent<Camera>(world, cameraData);
	}
}

void CreateCameraCommand::undo()
{
	Entity temp(entityData);

	world->latentDestroyEntity(temp.entityId);
}

CreateLightCommand::CreateLightCommand(World* world)
{
	this->world = world;
}

void CreateLightCommand::execute()
{
	if (entityData.size() == 0) {
		Entity* entity = world->createEntity();
		Transform* transform = entity->addComponent<Transform>(world);
		Light* light = entity->addComponent<Light>(world);
		entityData = entity->serialize();
		transformData = transform->serialize();
		lightData = light->serialize();
	}
	else {
		Entity* entity = world->createEntity(entityData);
		Transform* transform = entity->addComponent<Transform>(world, transformData);
		Camera* camera = entity->addComponent<Camera>(world, lightData);
	}
}

void CreateLightCommand::undo()
{
	Entity temp(entityData);

	world->latentDestroyEntity(temp.entityId);
}

CreateCubeCommand::CreateCubeCommand(World* world)
{
	this->world = world;
}

void CreateCubeCommand::execute()
{
	if (entityData.size() == 0) {
		Entity* entity = world->createEntity();
		Transform* transform = entity->addComponent<Transform>(world);
		BoxCollider* collider = entity->addComponent<BoxCollider>(world);
		MeshRenderer* meshRenderer = entity->addComponent<MeshRenderer>(world);
		entityData = entity->serialize();
		transformData = transform->serialize();
		boxColliderData = collider->serialize();
		meshRendererData = meshRenderer->serialize();
	}
	else {
		Entity* entity = world->createEntity(entityData);
		Transform* transform = entity->addComponent<Transform>(world, transformData);
		BoxCollider* collider = entity->addComponent<BoxCollider>(world, boxColliderData);
		MeshRenderer* meshRenderer = entity->addComponent<MeshRenderer>(world, meshRendererData);
	}
}

void CreateCubeCommand::undo()
{
	Entity temp(entityData);

	world->latentDestroyEntity(temp.entityId);
}


CreateSphereCommand::CreateSphereCommand(World* world)
{
	this->world = world;
}

void CreateSphereCommand::execute()
{
	if (entityData.size() == 0) {
		Entity* entity = world->createEntity();
		Transform* transform = entity->addComponent<Transform>(world);
		SphereCollider* collider = entity->addComponent<SphereCollider>(world);
		MeshRenderer* meshRenderer = entity->addComponent<MeshRenderer>(world);
		entityData = entity->serialize();
		transformData = transform->serialize();
		sphereColliderData = collider->serialize();
		meshRendererData = meshRenderer->serialize();
	}
	else {
		Entity* entity = world->createEntity(entityData);
		Transform* transform = entity->addComponent<Transform>(world, transformData);
		SphereCollider* collider = entity->addComponent<SphereCollider>(world, sphereColliderData);
		MeshRenderer* meshRenderer = entity->addComponent<MeshRenderer>(world, meshRendererData);
	}
}

void CreateSphereCommand::undo()
{
	Entity temp(entityData);

	world->latentDestroyEntity(temp.entityId);
}


DestroyEntityCommand::DestroyEntityCommand(World* world, PhysicsEngine::Guid entityId)
{
	this->world = world;

	std::vector<std::pair<Guid, int>> componentsOnEntity = world->getComponentsOnEntity(entityId);
	for (size_t i = 0; i < componentsOnEntity.size(); i++) {
		Guid componentId = componentsOnEntity[i].first;
		int componentType = componentsOnEntity[i].second;

		Component* component = NULL;
		if (componentType < 20) {
			// component = getComponentInternal(world, componentType);
		}
		else {
			// component = getComponent(world, componentType);
		}

		std::vector<char> componentData = component->serialize();

		std::pair<int, std::vector<char>> pair = std::make_pair(componentType, componentData);

		components.push_back(pair);
	}
}

void DestroyEntityCommand::execute()
{

}

void DestroyEntityCommand::undo()
{

}