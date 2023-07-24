#ifndef CAPSULECOLLIDER_H__
#define CAPSULECOLLIDER_H__

#define GLM_FORCE_RADIANS

#include "glm/glm.hpp"

#include "../core/SerializationEnums.h"
#include "../core/Guid.h"
#include "../core/Id.h"
#include "../core/Capsule.h"
#include "../core/AABB.h"

#include "ComponentEnums.h"

namespace PhysicsEngine
{
class World;

class CapsuleCollider
{
  private:
    Guid mGuid;
    Id mId;
    Guid mEntityGuid;

    World *mWorld;

  public:
    HideFlag mHide;
    bool mEnabled;

    Capsule mCapsule;

  public:
    CapsuleCollider(World *world, const Id &id);
    CapsuleCollider(World *world, const Guid &guid, const Id &id);
    ~CapsuleCollider();

    void serialize(YAML::Node &out) const;
    void deserialize(const YAML::Node &in);

    int getType() const;
    std::string getObjectName() const;

    Guid getEntityGuid() const;
    Guid getGuid() const;
    Id getId() const;

    bool intersect(AABB aabb) const;

  private:
    friend class Scene;
};

} // namespace PhysicsEngine

#endif