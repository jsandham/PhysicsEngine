#ifndef TERRAINSYSTEM_H__
#define TERRAINSYSTEM_H__

#include <vector>

#include "../core/SerializationEnums.h"
#include "../core/Guid.h"
#include "../core/Id.h"
#include "../core/Input.h"
#include "../core/Time.h"

#include "../components/Camera.h"

namespace PhysicsEngine
{
class World;

class TerrainSystem
{
  private:
    Guid mGuid;
    Id mId;
    World* mWorld;

  public:
    HideFlag mHide;
    bool mEnabled;

  public:
    TerrainSystem(World *world, const Id &id);
    TerrainSystem(World *world, const Guid &guid, const Id &id);
    ~TerrainSystem();

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