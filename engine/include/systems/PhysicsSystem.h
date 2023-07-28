#ifndef PHYSICSSYSTEM_H__
#define PHYSICSSYSTEM_H__

#include <vector>

#include "../core/SerializationEnums.h"
#include "../core/Guid.h"
#include "../core/Id.h"
#include "../core/Input.h"
#include "../core/Time.h"

#include "../components/Collider.h"
#include "../components/Rigidbody.h"

namespace PhysicsEngine
{
class World; 

class PhysicsSystem
{
  private:
    Guid mGuid;
    Id mId;
    World* mWorld;

    std::vector<Rigidbody *> mRigidbodies;

    float mTimestep;
    float mGravity;

  public:
    HideFlag mHide;
    bool mEnabled;

  public:
    PhysicsSystem(World *world, const Id &id);
    PhysicsSystem(World *world, const Guid &guid, const Id &id);
    ~PhysicsSystem();

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