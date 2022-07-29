#ifndef ENTITY_H__
#define ENTITY_H__

#include <string>
#include <vector>

#include "Object.h"
#include "Types.h"

namespace PhysicsEngine
{
class Entity : public Object
{
  private:
    std::string mName;

  public:
    bool mDoNotDestroy;
    bool mEnabled;

  public:
    Entity(World *world, const Id &id);
    Entity(World *world, const Guid& guid, const Id& id);
    ~Entity();

    virtual void serialize(YAML::Node &out) const override;
    virtual void deserialize(const YAML::Node &in) override;

    virtual int getType() const override;
    virtual std::string getObjectName() const override;

    void latentDestroy();
    void immediateDestroy();

    template <typename T> T *addComponent()
    {
        return mWorld->getActiveScene()->addComponent<T>(getGuid());
    }

    template <typename T> T *getComponent() const
    {
        return mWorld->getActiveScene()->getComponent<T>(getGuid());
    }

    std::vector<std::pair<Guid, int>> getComponentsOnEntity() const;

    std::string getName() const;
    void setName(const std::string &name);

  private:
    friend class Scene;
};

template <typename T> struct EntityType
{
    static constexpr int type = PhysicsEngine::INVALID_TYPE;
};

template <typename> struct IsEntityInternal
{
    static constexpr bool value = false;
};

template <> struct IsEntityInternal<Entity>
{
    static constexpr bool value = true;
};
} // namespace PhysicsEngine

#endif