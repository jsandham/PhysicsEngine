#ifndef __MESHCOLLIDER_H__
#define __MESHCOLLIDER_H__

#include <vector>

#include "Collider.h"

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"

namespace PhysicsEngine
{
class MeshCollider : public Collider
{
  public:
    Guid mMeshId;

  public:
    MeshCollider();
    MeshCollider(Guid id);
    ~MeshCollider();

    virtual void serialize(std::ostream& out) const;
    virtual void deserialize(std::istream& in);

    bool intersect(AABB aabb) const;
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