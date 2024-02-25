#ifndef BOXCOLLIDER_H__
#define BOXCOLLIDER_H__

#include "../core/glm.h"
#include "../core/SerializationEnums.h"
#include "../core/AABB.h"
#include "../core/Guid.h"
#include "../core/Id.h"

#include "ComponentYaml.h"
#include "ComponentEnums.h"

namespace PhysicsEngine
{
class World;

struct BoxColliderData
{
    AABB mAABB;

    void serialize(YAML::Node &out) const;
    void deserialize(const YAML::Node &in);
};

class BoxCollider
{
  private:
    Guid mGuid;
    Id mId;
    Guid mEntityGuid;

    World *mWorld;

  public:
    HideFlag mHide;
    bool mEnabled;

    AABB mAABB;

  public:
    BoxCollider(World *world, const Id &id);
    BoxCollider(World *world, const Guid &guid, const Id &id);

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