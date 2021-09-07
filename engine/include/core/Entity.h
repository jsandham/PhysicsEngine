#ifndef ENTITY_H__
#define ENTITY_H__

#include <string>
#include <vector>

#include "Guid.h"
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
    Entity(World *world);
    Entity(World *world, Guid id);
    ~Entity();

    virtual void serialize(YAML::Node &out) const override;
    virtual void deserialize(const YAML::Node &in) override;

    virtual int getType() const override;
    virtual std::string getObjectName() const override;

    void latentDestroy();
    void immediateDestroy();

    template <typename T> T *addComponent()
    {
        return mWorld->addComponent<T>(getId());
    }

    template <typename T> T *getComponent() const
    {
        return mWorld->getComponent<T>(getId());
    }

    std::vector<std::pair<Guid, int>> getComponentsOnEntity() const;

    std::string getName() const;
    void setName(const std::string &name);

  private:
    friend class World;
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