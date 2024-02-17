#ifndef CLEANUPSYSTEM_H__
#define CLEANUPSYSTEM_H__

#include <vector>

#include "../core/SerializationEnums.h"
#include "../core/Guid.h"
#include "../core/Id.h"

namespace PhysicsEngine
{
class World;

class CleanUpSystem
{
  private:
    Guid mGuid;
    Id mId;
    World* mWorld;

  public:
    HideFlag mHide;
    bool mEnabled;

  public:
    CleanUpSystem(World *world, const Id &id);
    CleanUpSystem(World *world, const Guid &guid, const Id &id);

    void serialize(YAML::Node &out) const;
    void deserialize(const YAML::Node &in);

    int getType() const;
    std::string getObjectName() const;

    Guid getGuid() const;
    Id getId() const;

    void init(World *world);
    void update();
};

} // namespace PhysicsEngine

#endif