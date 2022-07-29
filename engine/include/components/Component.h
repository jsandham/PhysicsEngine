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
    Guid mEntityGuid;

  public:
    Component(World *world, const Id &id);
    Component(World *world, const Guid& guid, const Id& id);
    ~Component();

    virtual void serialize(YAML::Node &out) const override;
    virtual void deserialize(const YAML::Node &in) override;

    Entity *getEntity() const;

    template <typename T> void latentDestroy()
    {
        mWorld->getActiveScene()->latentDestroyComponent(mEntityGuid, getGuid(), ComponentType<T>::type);
    }

    template <typename T> void immediateDestroy()
    {
        mWorld->getActiveScene()->immediateDestroyComponent(mEntityGuid, getGuid(), ComponentType<T>::type);
    }

    template <typename T> T *getComponent() const
    {
        return mWorld->getActiveScene()->getComponent<T>(mEntityGuid);
    }

    Guid getEntityGuid() const;

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