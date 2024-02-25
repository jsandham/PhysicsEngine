#ifndef SPHERECOLLIDER_H__
#define SPHERECOLLIDER_H__

#include "../core/glm.h"
#include "../core/SerializationEnums.h"
#include "../core/Guid.h"
#include "../core/Id.h"
#include "../core/Sphere.h"
#include "../core/AABB.h"

#include "ComponentYaml.h"
#include "ComponentEnums.h"

namespace PhysicsEngine
{
class World;

struct SphereColliderData
{
    Sphere mSphere;

    void serialize(YAML::Node &out) const;
    void deserialize(const YAML::Node &in);
};

class SphereCollider
{
  private:
    Guid mGuid;
    Id mId;
    Guid mEntityGuid;

    World *mWorld;

  public:
    HideFlag mHide;
    bool mEnabled;

    Sphere mSphere;

  public:
    SphereCollider(World *world, const Id &id);
    SphereCollider(World *world, const Guid &guid, const Id &id);

    void serialize(YAML::Node &out) const;
    void deserialize(const YAML::Node &in);

    int getType() const;
    std::string getObjectName() const;

    Guid getEntityGuid() const;
    Guid getGuid() const;
    Id getId() const;

    bool intersect(AABB aabb) const;

    std::vector<float> getLines() const;

    template <typename T> T *getComponent() const
    {
        return mWorld->getActiveScene()->getComponent<T>(mEntityGuid);
    }

  private:
    friend class Scene;
};

} // namespace PhysicsEngine

#endif