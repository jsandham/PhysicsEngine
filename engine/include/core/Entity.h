#ifndef __ENTITY_H__
#define __ENTITY_H__

#include <string>
#include <vector>

#include "Object.h"
#include "Guid.h"
#include "Types.h"

namespace PhysicsEngine
{
#pragma pack(push, 1)
struct EntityHeader
{
    Guid mEntityId;
    char mEntityName[64];
    uint8_t mDoNotDestroy;
};
#pragma pack(pop)

#pragma pack(push, 1)
struct EntityHeader1
{
    char mEntityName[64];
    uint8_t mDoNotDestroy;
};
#pragma pack(pop)

class World;

class Entity : public Object
{
  private:
    std::string mEntityName;

  public:
    bool mDoNotDestroy;

  public:
    Entity();
    Entity(Guid id);
    ~Entity();

    std::vector<char> serialize() const;
    std::vector<char> serialize(const Guid &entityId) const;
    void deserialize(const std::vector<char> &data);

    void latentDestroy(World *world);
    void immediateDestroy(World *world);

    template <typename T> T *addComponent(World *world)
    {
        return world->addComponent<T>(mId);
    }

    template <typename T> T *addComponent(World *world, std::vector<char> data)
    {
        return world->addComponent<T>(data);
    }

    template <typename T> T *getComponent(World *world)
    {
        return world->getComponent<T>(mId);
    }

    std::vector<std::pair<Guid, int>> getComponentsOnEntity(World *world);

    std::string getName() const;
    void setName(const std::string &name);

  private:
    friend class World;
};

template <typename T> struct EntityType
{
    static constexpr int type = PhysicsEngine::INVALID_TYPE;
};
template <typename T> struct IsEntity
{
    static constexpr bool value = false;
};
template <typename> struct IsEntityInternal
{
    static constexpr bool value = false;
};

template <> struct EntityType<Entity>
{
    static constexpr int type = PhysicsEngine::ENTITY_TYPE;
};
template <> struct IsEntity<Entity>
{
    static constexpr bool value = true;
};
template <> struct IsEntityInternal<Entity>
{
    static constexpr bool value = true;
};
} // namespace PhysicsEngine

#endif