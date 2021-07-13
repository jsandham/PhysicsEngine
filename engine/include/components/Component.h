#ifndef COMPONENT_H__
#define COMPONENT_H__

#include "../core/Guid.h"
#include "../core/Object.h"
#include "../core/Types.h"

namespace PhysicsEngine
{
class Entity;
class World;

class Component : public Object
{
  protected:
    Guid mEntityId;

  public:
    Component();
    Component(Guid id);
    ~Component();

    virtual void serialize(YAML::Node &out) const override;
    virtual void deserialize(const YAML::Node &in) override;

    Entity *getEntity(const World *world) const;

    template <typename T> void latentDestroy(World *world)
    {
        world->latentDestroyComponent(entityId, componentId, getInstanceType<T>());
    }

    template <typename T> void immediateDestroy(World *world)
    {
        world->immediateDestroyComponent(entityId, componentId, getInstanceType<T>());
    }

    template <typename T> T *getComponent(const World *world) const
    {
        const Entity *entity = getEntity(world);

        return entity->getComponent<T>(world);
    }

    Guid getEntityId() const;

    static bool isInternal(int type);

  private:
    friend class World;
};

template <typename T> struct ComponentType
{
    static constexpr int type = PhysicsEngine::INVALID_TYPE;
};

template <typename T> struct IsComponentInternal
{
    static constexpr bool value = false;
};

} // namespace PhysicsEngine

#endif