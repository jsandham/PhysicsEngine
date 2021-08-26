#include "../include/EditorCommands.h"

#include "components/BoxCollider.h"
#include "components/Light.h"
#include "components/MeshRenderer.h"
#include "components/SphereCollider.h"
#include "core/Entity.h"

using namespace PhysicsEditor;
using namespace PhysicsEngine;

// Hierarchy commands

CreateEntityCommand::CreateEntityCommand(World *world, bool *saveStatePtr)
{
    this->world = world;
    this->saveStatePtr = saveStatePtr;
    this->oldSaveState = *saveStatePtr;
}

void CreateEntityCommand::execute()
{
    if (entityData.size() == 0)
    {
        Entity *entity = world->createEntity();
        Transform *transform = entity->addComponent<Transform>();
        //entityData = entity->serialize();
        //transformData = transform->serialize();
    }
    else
    {
        //Entity *entity = world->createEntity(entityData);
        //Transform *transform = entity->addComponent<Transform>(world, transformData);
    }

    *saveStatePtr = true;
}

void CreateEntityCommand::undo()
{
    //Entity temp;
    //temp.deserialize(entityData);

    //world->latentDestroyEntity(temp.getId());

    //*saveStatePtr = oldSaveState;
}

CreateCameraCommand::CreateCameraCommand(World *world, bool *saveStatePtr)
{
    this->world = world;
    this->saveStatePtr = saveStatePtr;
    this->oldSaveState = *saveStatePtr;
}

void CreateCameraCommand::execute()
{
    if (entityData.size() == 0)
    {
        Entity *entity = world->createEntity();
        Transform *transform = entity->addComponent<Transform>();
        Camera *camera = entity->addComponent<Camera>();
        //entityData = entity->serialize();
        //transformData = transform->serialize();
        //cameraData = camera->serialize();
    }
    else
    {
        //Entity *entity = world->createEntity(entityData);
        //Transform *transform = entity->addComponent<Transform>(world, transformData);
        //Camera *camera = entity->addComponent<Camera>(world, cameraData);
    }

    *saveStatePtr = true;
}

void CreateCameraCommand::undo()
{
    //Entity temp;
    //temp.deserialize(entityData);

    //world->latentDestroyEntity(temp.getId());

    *saveStatePtr = oldSaveState;
}

CreateLightCommand::CreateLightCommand(World *world, bool *saveStatePtr)
{
    this->world = world;
    this->saveStatePtr = saveStatePtr;
    this->oldSaveState = *saveStatePtr;
}

void CreateLightCommand::execute()
{
    if (entityData.size() == 0)
    {
        Entity *entity = world->createEntity();
        Transform *transform = entity->addComponent<Transform>();
        Light *light = entity->addComponent<Light>();
        //entityData = entity->serialize();
        //transformData = transform->serialize();
        //lightData = light->serialize();
    }
    else
    {
        //Entity *entity = world->createEntity(entityData);
        //Transform *transform = entity->addComponent<Transform>(world, transformData);
        //Camera *camera = entity->addComponent<Camera>(world, lightData);
    }

    *saveStatePtr = true;
}

void CreateLightCommand::undo()
{
    //Entity temp;
    //temp.deserialize(entityData);

    //world->latentDestroyEntity(temp.getId());

    *saveStatePtr = oldSaveState;
}

CreatePlaneCommand::CreatePlaneCommand(World *world, bool *saveStatePtr)
{
    this->world = world;
    this->saveStatePtr = saveStatePtr;
    this->oldSaveState = *saveStatePtr;
}

void CreatePlaneCommand::execute()
{
    if (entityData.size() == 0)
    {
        Entity *entity = world->createEntity();
        Transform *transform = entity->addComponent<Transform>();
        MeshRenderer *meshRenderer = entity->addComponent<MeshRenderer>();
        meshRenderer->setMesh(world->getAssetId("data\\meshes\\plane.mesh"));
        meshRenderer->setMaterial(world->getAssetId("data\\materials\\color.material"));

        //entityData = entity->serialize();
        //transformData = transform->serialize();
        //meshRendererData = meshRenderer->serialize();
    }
    else
    {
        //Entity *entity = world->createEntity(entityData);
        //Transform *transform = entity->addComponent<Transform>(world, transformData);
        //MeshRenderer *meshRenderer = entity->addComponent<MeshRenderer>(world, meshRendererData);
    }

    *saveStatePtr = true;
}

void CreatePlaneCommand::undo()
{
    //Entity temp;
    //temp.deserialize(entityData);

    //world->latentDestroyEntity(temp.getId());

    *saveStatePtr = oldSaveState;
}

CreateCubeCommand::CreateCubeCommand(World *world, bool *saveStatePtr)
{
    this->world = world;
    this->saveStatePtr = saveStatePtr;
    this->oldSaveState = *saveStatePtr;
}

void CreateCubeCommand::execute()
{
    if (entityData.size() == 0)
    {
        Entity *entity = world->createEntity();
        Transform *transform = entity->addComponent<Transform>();
        BoxCollider *collider = entity->addComponent<BoxCollider>();
        MeshRenderer *meshRenderer = entity->addComponent<MeshRenderer>();
        meshRenderer->setMesh(world->getAssetId("data\\meshes\\cube.mesh"));
        meshRenderer->setMaterial(world->getAssetId("data\\materials\\color.material"));

        //entityData = entity->serialize();
        //transformData = transform->serialize();
        //boxColliderData = collider->serialize();
        //meshRendererData = meshRenderer->serialize();
    }
    else
    {
        //Entity *entity = world->createEntity(entityData);
        //Transform *transform = entity->addComponent<Transform>(world, transformData);
        //BoxCollider *collider = entity->addComponent<BoxCollider>(world, boxColliderData);
        //MeshRenderer *meshRenderer = entity->addComponent<MeshRenderer>(world, meshRendererData);
    }

    *saveStatePtr = true;
}

void CreateCubeCommand::undo()
{
    //Entity temp;
    //temp.deserialize(entityData);

    //world->latentDestroyEntity(temp.getId());

    *saveStatePtr = oldSaveState;
}

CreateSphereCommand::CreateSphereCommand(World *world, bool *saveStatePtr)
{
    this->world = world;
    this->saveStatePtr = saveStatePtr;
    this->oldSaveState = *saveStatePtr;
}

void CreateSphereCommand::execute()
{
    if (entityData.size() == 0)
    {
        Entity *entity = world->createEntity();
        Transform *transform = entity->addComponent<Transform>();
        SphereCollider *collider = entity->addComponent<SphereCollider>();
        MeshRenderer *meshRenderer = entity->addComponent<MeshRenderer>();
        meshRenderer->setMesh(world->getAssetId("data\\meshes\\sphere.mesh"));
        /*meshRenderer->setMaterial(world->getAssetId("data\\materials\\color.material"));*/
        meshRenderer->setMaterial(world->getAssetId("data\\materials\\default.material"));

        //entityData = entity->serialize();
        //transformData = transform->serialize();
        //sphereColliderData = collider->serialize();
        //meshRendererData = meshRenderer->serialize();
    }
    else
    {
        //Entity *entity = world->createEntity(entityData);
        //Transform *transform = entity->addComponent<Transform>(world, transformData);
        //SphereCollider *collider = entity->addComponent<SphereCollider>(world, sphereColliderData);
        //MeshRenderer *meshRenderer = entity->addComponent<MeshRenderer>(world, meshRendererData);
    }

    *saveStatePtr = true;
}

void CreateSphereCommand::undo()
{
    //Entity temp;
    //temp.deserialize(entityData);

    //world->latentDestroyEntity(temp.getId());

    *saveStatePtr = oldSaveState;
}

DestroyEntityCommand::DestroyEntityCommand(World *world, PhysicsEngine::Guid entityId, bool *saveStatePtr)
{
    this->world = world;
    this->saveStatePtr = saveStatePtr;
    this->oldSaveState = *saveStatePtr;

    std::vector<std::pair<Guid, int>> componentsOnEntity = world->getComponentsOnEntity(entityId);
    for (size_t i = 0; i < componentsOnEntity.size(); i++)
    {
        Guid componentId = componentsOnEntity[i].first;
        int componentType = componentsOnEntity[i].second;

        Component *component = NULL;
        if (Component::isInternal(componentType))
        {
            // component = getComponentInternal(world, componentType);
        }
        else
        {
            // component = getComponent(world, componentType);
        }

        //std::vector<char> componentData = component->serialize();

        //std::pair<int, std::vector<char>> pair = std::make_pair(componentType, componentData);

        //components.push_back(pair);
    }
}

void DestroyEntityCommand::execute()
{
    *saveStatePtr = true;
}

void DestroyEntityCommand::undo()
{
    *saveStatePtr = oldSaveState;
}