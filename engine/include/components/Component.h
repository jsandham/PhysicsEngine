#ifndef __COMPONENT_H__
#define __COMPONENT_H__

#include "../core/Object.h"
#include "../core/Guid.h"
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

    virtual std::vector<char> serialize(const Guid &componentId, const Guid &entityId) const = 0;
    virtual std::vector<char> serialize() const = 0;
    virtual void deserialize(const std::vector<char>& data) = 0;

    Entity *getEntity(World *world) const;

    template <typename T> void latentDestroy(World *world)
    {
        world->latentDestroyComponent(entityId, componentId, getInstanceType<T>());
    }

    template <typename T> void immediateDestroy(World *world)
    {
        world->immediateDestroyComponent(entityId, componentId, getInstanceType<T>());
    }

    template <typename T> T *getComponent(World *world)
    {
        Entity *entity = getEntity(world);

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