#ifndef MESHCOLLIDER_H__
#define MESHCOLLIDER_H__

#include "Collider.h"

#define GLM_FORCE_RADIANS

#include "glm/glm.hpp"

namespace PhysicsEngine
{
class MeshCollider : public Collider
{
  public:
    Guid mMeshId;

  public:
    MeshCollider(World *world);
    MeshCollider(World *world, const Guid& id);
    ~MeshCollider();

    virtual void serialize(YAML::Node &out) const override;
    virtual void deserialize(const YAML::Node &in) override;

    virtual int getType() const override;
    virtual std::string getObjectName() const override;

    bool intersect(AABB aabb) const override;
};

template <> struct ComponentType<MeshCollider>
{
    static constexpr int type = PhysicsEngine::MESHCOLLIDER_TYPE;
};

template <> struct IsComponentInternal<MeshCollider>
{
    static constexpr bool value = true;
};
} // namespace PhysicsEngine

#endif