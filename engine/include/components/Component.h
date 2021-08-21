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
    Component(World* world);
    Component(World* world, Guid id);
    ~Component();

    virtual void serialize(YAML::Node &out) const override;
    virtual void deserialize(const YAML::Node &in) override;

    Entity* getEntity() const;

    template <typename T> void latentDestroy()
    {
        mWorld->latentDestroyComponent(mEntityId, getId(), ComponentType<T>::type);
    }

    template <typename T> void immediateDestroy()
    {
        mWorld->immediateDestroyComponent(mEntityId, getId(), ComponentType<T>::type);
    }

    template <typename T> T* getComponent() const
    {
        const Entity* entity = getEntity();

        if (entity != nullptr)
        {
            return entity->getComponent<T>();
        }

        return nullptr;
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