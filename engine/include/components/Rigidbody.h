#ifndef RIGIDBODY_H__
#define RIGIDBODY_H__

#include "../core/glm.h"
#include "../core/SerializationEnums.h"
#include "../core/Guid.h"
#include "../core/Id.h"

#include "ComponentEnums.h"

namespace PhysicsEngine
{
class World;

struct RigidbodyData
{
    glm::vec3 mVelocity;
    glm::vec3 mAngularVelocity;
    glm::vec3 mCentreOfMass;
    glm::mat3 mInertiaTensor;

    // leap-frog
    glm::vec3 mHalfVelocity;

    float mMass;
    float mDrag;
    float mAngularDrag;
    bool mUseGravity;

    RigidbodyData();

    void serialize(YAML::Node &out) const;
    void deserialize(const YAML::Node &in);
};

class Rigidbody
{ 
  private:
    Guid mGuid;
    Id mId;
    Guid mEntityGuid;

    World *mWorld;

  public:
    HideFlag mHide;
    bool mEnabled;




    glm::vec3 mVelocity;
    glm::vec3 mAngularVelocity;
    glm::vec3 mCentreOfMass;
    glm::mat3 mInertiaTensor;

    // leap-frog
    glm::vec3 mHalfVelocity;

    float mMass;
    float mDrag;
    float mAngularDrag;
    bool mUseGravity;

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

    template <typename T> T *getComponent() const
    {
        return mWorld->getActiveScene()->getComponent<T>(mEntityGuid);
    }

  private:
    friend class Scene;
};
} // namespace PhysicsEngine

#endif