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
	Entity* entity = world->createEntity();
	entityId = entity->entityId;
}

void CreateEntityCommand::undo()
{
	world->latentDestroyEntity(entityId);
}


CreateCameraCommand::CreateCameraCommand(World* world)
{
	this->world = world;
}

void CreateCameraCommand::execute()
{
	Entity* entity = world->createEntity();
	entityId = entity->entityId;

	entity->addComponent<Camera>(world);
}

void CreateCameraCommand::undo()
{
	world->latentDestroyEntity(entityId);
}


CreateCubeCommand::CreateCubeCommand(World* world)
{
	this->world = world;
}

void CreateCubeCommand::execute()
{
	Entity* entity = world->createEntity();
	entityId = entity->entityId;

	entity->addComponent<BoxCollider>(world);
	entity->addComponent<MeshRenderer>(world);
}

void CreateCubeCommand::undo()
{
	world->latentDestroyEntity(entityId);
}


CreateSphereCommand::CreateSphereCommand(World* world)
{
	this->world = world;
}

void CreateSphereCommand::execute()
{
	Entity* entity = world->createEntity();
	entityId = entity->entityId;

	entity->addComponent<SphereCollider>(world);
	entity->addComponent<MeshRenderer>(world);
}

void CreateSphereCommand::undo()
{
	world->latentDestroyEntity(entityId);
}


CreateLightCommand::CreateLightCommand(World* world)
{
	this->world = world;
}

void CreateLightCommand::execute()
{
	Entity* entity = world->createEntity();
	entityId = entity->entityId;

	entity->addComponent<Light>(world);
}

void CreateLightCommand::undo()
{
	world->latentDestroyEntity(entityId);
}