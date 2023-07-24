#ifndef RIGIDBODY_H__
#define RIGIDBODY_H__

#define GLM_FORCE_RADIANS

#include "glm/glm.hpp"

#include "../core/SerializationEnums.h"
#include "../core/Guid.h"
#include "../core/Id.h"

#include "ComponentEnums.h"

namespace PhysicsEngine
{
class World;

class Rigidbody
{ 
  private:
    Guid mGuid;
    Id mId;
    Guid mEntityGuid;

    World *mWorld;

  public:
    HideFlag mHide;

    float mMass;
    float mDrag;
    float mAngularDrag;
    bool mUseGravity;
    bool mEnabled;

    glm::vec3 mVelocity;
    glm::vec3 mAngularVelocity;
    glm::vec3 mCentreOfMass;
    glm::mat3 mInertiaTensor;

    // leap-frog
    glm::vec3 mHalfVelocity;

  public:
    Rigidbody(World *world, const Id &id);
    Rigidbody(World *world, const Guid &guid, const Id &id);
    ~Rigidbody();

    void serialize(YAML::Node &out) const;
    void deserialize(const YAML::Node &in);

    int getType() const;
    std::string getObjectName() const;

    Guid getEntityGuid() const;
    Guid getGuid() const;
    Id getId() const;

  private:
    friend class Scene;
};
} // namespace PhysicsEngine

#endif