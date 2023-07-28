#ifndef MESHCOLLIDER_H__
#define MESHCOLLIDER_H__

#define GLM_FORCE_RADIANS

#include "glm/glm.hpp"

#include "../core/SerializationEnums.h"
#include "../core/Guid.h"
#include "../core/Id.h"
#include "../core/AABB.h"

#include "ComponentEnums.h"

namespace PhysicsEngine
{
class World;

class MeshCollider
{
  private:
    Guid mGuid;
    Id mId;
    Guid mEntityGuid;

    World *mWorld;

  public:
    HideFlag mHide;
    bool mEnabled;

    Guid mMeshId;

  public:
    MeshCollider(World *world, const Id &id);
    MeshCollider(World *world, const Guid &guid, const Id &id);
    ~MeshCollider();

    void serialize(YAML::Node &out) const;
    void deserialize(const YAML::Node &in);

    int getType() const;
    std::string getObjectName() const;

    Guid getEntityGuid() const;
    Guid getGuid() const;
    Id getId() const;

    bool intersect(AABB aabb) const;

    template <typename T> T *getComponent() const
    {
        return mWorld->getActiveScene()->getComponent<T>(mEntityGuid);
    }

  private:
    friend class Scene;
};

} // namespace PhysicsEngine

#endif