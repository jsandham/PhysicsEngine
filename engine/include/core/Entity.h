#ifndef ENTITY_H__
#define ENTITY_H__

#include <string>
#include <vector>

#include "Guid.h"
#include "Object.h"
#include "Types.h"

namespace PhysicsEngine
{
class World;

class Entity : public Object
{
  private:
    std::string mName;

  public:
    bool mDoNotDestroy;

  public:
    Entity();
    Entity(Guid id);
    ~Entity();

    virtual void serialize(std::ostream &out) const override;
    virtual void deserialize(std::istream &in) override;
    virtual void serialize(YAML::Node& out) const override;
    virtual void deserialize(const YAML::Node& in) override;

    virtual int getType() const override;
    virtual std::string getObjectName() const override;

    void latentDestroy(World *world);
    void immediateDestroy(World *world);

    template <typename T> T *addComponent(World *world)
    {
        return world->addComponent<T>(getId());
    }

    template <typename T> T *addComponent(World *world, std::vector<char> data)
    {
        return world->addComponent<T>(data);
    }

    template <typename T> T *getComponent(const World *world) const
    {
        return world->getComponent<T>(getId());
    }

    std::vector<std::pair<Guid, int>> getComponentsOnEntity(const World *world) const;

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