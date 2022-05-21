#ifndef COMPONENT_H__
#define COMPONENT_H__

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
    Component(World *world);
    Component(World *world, const Guid& id);
    ~Component();

    virtual void serialize(YAML::Node &out) const override;
    virtual void deserialize(const YAML::Node &in) override;

    Entity *getEntity() const;

    template <typename T> void latentDestroy()
    {
        mWorld->getActiveScene()->latentDestroyComponent(mEntityId, getId(), ComponentType<T>::type);
    }

    template <typename T> void immediateDestroy()
    {
        mWorld->getActiveScene()->immediateDestroyComponent(mEntityId, getId(), ComponentType<T>::type);
    }

    template <typename T> T *getComponent() const
    {
        return mWorld->getActiveScene()->getComponent<T>(mEntityId);
    }

    Guid getEntityId() const;

    static bool isInternal(int type);

  private:
    friend class Scene;
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