#ifndef __MESHCOLLIDER_H__
#define __MESHCOLLIDER_H__

#include <vector>

#include "Collider.h"

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"

namespace PhysicsEngine
{
#pragma pack(push, 1)
struct MeshColliderHeader
{
    Guid mComponentId;
    Guid mEntityId;
    Guid mMeshId;
};
#pragma pack(pop)

class MeshCollider : public Collider
{
  public:
    Guid mMeshId;

  public:
    MeshCollider();
    MeshCollider(Guid id);
    ~MeshCollider();

    std::vector<char> serialize() const;
    std::vector<char> serialize(const Guid &componentId, const Guid &entityId) const;
    void deserialize(const std::vector<char> &data);

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