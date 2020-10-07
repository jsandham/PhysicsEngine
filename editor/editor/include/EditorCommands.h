#ifndef __EDITOR_COMMANDS_H__
#define __EDITOR_COMMANDS_H__

#include "Command.h"

#include "components/Rigidbody.h"
#include "components/Transform.h"
#include "core/Guid.h"
#include "core/World.h"

namespace PhysicsEditor
{
// Hierarchy commands

class CreateEntityCommand : public Command
{
  private:
    PhysicsEngine::World *world;
    std::vector<char> entityData;
    std::vector<char> transformData;

    bool *saveStatePtr;
    bool oldSaveState;

  public:
    CreateEntityCommand(PhysicsEngine::World *world, bool *saveStatePtr);

    void execute() override;
    void undo() override;
};

class CreateCameraCommand : public Command
{
  private:
    PhysicsEngine::World *world;
    std::vector<char> entityData;
    std::vector<char> transformData;
    std::vector<char> cameraData;

    bool *saveStatePtr;
    bool oldSaveState;

  public:
    CreateCameraCommand(PhysicsEngine::World *world, bool *saveStatePtr);

    void execute() override;
    void undo() override;
};

class CreateLightCommand : public Command
{
  private:
    PhysicsEngine::World *world;
    std::vector<char> entityData;
    std::vector<char> transformData;
    std::vector<char> lightData;

    bool *saveStatePtr;
    bool oldSaveState;

  public:
    CreateLightCommand(PhysicsEngine::World *world, bool *saveStatePtr);

    void execute() override;
    void undo() override;
};

class CreateCubeCommand : public Command
{
  private:
    PhysicsEngine::World *world;
    std::vector<char> entityData;
    std::vector<char> transformData;
    std::vector<char> boxColliderData;
    std::vector<char> meshRendererData;

    bool *saveStatePtr;
    bool oldSaveState;

  public:
    CreateCubeCommand(PhysicsEngine::World *world, bool *saveStatePtr);

    void execute() override;
    void undo() override;
};

class CreateSphereCommand : public Command
{
  private:
    PhysicsEngine::World *world;
    std::vector<char> entityData;
    std::vector<char> transformData;
    std::vector<char> sphereColliderData;
    std::vector<char> meshRendererData;

    bool *saveStatePtr;
    bool oldSaveState;

  public:
    CreateSphereCommand(PhysicsEngine::World *world, bool *saveStatePtr);

    void execute() override;
    void undo() override;
};

class DestroyEntityCommand : public Command
{
  private:
    PhysicsEngine::World *world;
    std::vector<char> entityData;
    std::vector<std::pair<int, std::vector<char>>> components;

    bool *saveStatePtr;
    bool oldSaveState;

  public:
    DestroyEntityCommand(PhysicsEngine::World *world, PhysicsEngine::Guid entityId, bool *saveStatePtr);

    void execute() override;
    void undo() override;
};

// inspector commands

template <class T> class ChangePropertyCommand : public Command
{
  private:
    T *valuePtr;
    T oldValue;
    T newValue;

    bool *saveStatePtr;
    bool oldSaveState;

  public:
    ChangePropertyCommand(T *valuePtr, T newValue, bool *saveStatePtr)
    {
        this->valuePtr = valuePtr;
        this->oldValue = *valuePtr;
        this->newValue = newValue;

        this->saveStatePtr = saveStatePtr;
        this->oldSaveState = *saveStatePtr;
    }

    void execute()
    {
        *valuePtr = newValue;
        *saveStatePtr = true;
    }

    void undo()
    {
        *valuePtr = oldValue;
        *saveStatePtr = oldSaveState;
    }
};

template <class T> class AddComponentCommand : public Command
{
  private:
    PhysicsEngine::World *world;
    PhysicsEngine::Guid entityId;
    PhysicsEngine::Guid componentId;

    bool *saveStatePtr;
    bool oldSaveState;

  public:
    AddComponentCommand(PhysicsEngine::World *world, PhysicsEngine::Guid entityId, bool *saveStatePtr)
    {
        this->world = world;
        this->entityId = entityId;

        this->saveStatePtr = saveStatePtr;
        this->oldSaveState = *saveStatePtr;
    }

    void execute()
    {
        Entity *entity = world->getEntityById(entityId);
        T *component = entity->addComponent<T>(world);
        componentId = component->getId();

        *saveStatePtr = true;
    }

    void undo()
    {
        world->latentDestroyComponent(entityId, componentId, ComponentType<T>::type);

        *saveStatePtr = oldSaveState;
    }
};

// template<class T>
// class ChangeComponentValue
} // namespace PhysicsEditor

#endif
