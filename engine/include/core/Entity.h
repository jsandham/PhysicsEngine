#ifndef ENTITY_H__
#define ENTITY_H__

#include <string>
#include <vector>

#include "Guid.h"
#include "Id.h"
#include "Types.h"
#include "SerializationEnums.h"

namespace PhysicsEngine
{
class World;

class Entity
{
  private:
    Guid mGuid;
    Id mId;
    World *mWorld;

  public:
    HideFlag mHide;
    std::string mName;

    bool mDoNotDestroy;
    bool mEnabled;

  public:
    Entity(World *world, const Id &id);
    Entity(World *world, const Guid &guid, const Id &id);

    void serialize(YAML::Node &out) const;
    void deserialize(const YAML::Node &in);

    int getType() const;
    std::string getObjectName() const;

    Guid getGuid() const;
    Id getId() const;

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

  private:
    friend class Scene;
};

template <typename T> struct EntityType
{
    static constexpr int type = PhysicsEngine::INVALID_TYPE;
};

} // namespace PhysicsEngine

#endif