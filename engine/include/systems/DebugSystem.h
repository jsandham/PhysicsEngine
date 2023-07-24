#ifndef DEBUGSYSTEM_H__
#define DEBUGSYSTEM_H__

#include <vector>

#include "../core/SerializationEnums.h"
#include "../core/Guid.h"
#include "../core/Id.h"
#include "../core/Input.h"
#include "../core/Time.h"
#include "../core/Material.h"
#include "../core/Shader.h"

namespace PhysicsEngine
{
class World;

class DebugSystem
{
  private:
    Guid mGuid;
    Id mId;
    World* mWorld;

  public:
    HideFlag mHide;

  public:
    DebugSystem(World *world, const Id &id);
    DebugSystem(World *world, const Guid &guid, const Id &id);
    ~DebugSystem();

    void serialize(YAML::Node &out) const;
    void deserialize(const YAML::Node &in);

    int getType() const;
    std::string getObjectName() const;

    Guid getGuid() const;
    Id getId() const;

    void init(World *world);
    void update(const Input &input, const Time &time);
};

} // namespace PhysicsEngine

#endif