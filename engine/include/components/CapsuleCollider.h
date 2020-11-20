#ifndef __CAPSULECOLLIDER_H__
#define __CAPSULECOLLIDER_H__

#include <vector>

#include "Collider.h"

#include "../core/Capsule.h"

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"

namespace PhysicsEngine
{
#pragma pack(push, 1)
struct CapsuleColliderHeader
{
    Guid mComponentId;
    Guid mEntityId;
    Capsule mCapsule;
};
#pragma pack(pop)

class CapsuleCollider : public Collider
{
  public:
    Capsule mCapsule;

  public:
    CapsuleCollider();
    CapsuleCollider(const std::vector<char> &data);
    ~CapsuleCollider();

    std::vector<char> serialize() const;
    std::vector<char> serialize(const Guid &componentId, const Guid &entityId) const;
    void deserialize(const std::vector<char> &data);

    bool intersect(AABB aabb) const;
};

template <typename T> struct IsCapsuleCollider
{
    static constexpr bool value = false;
};

template <> struct ComponentType<CapsuleCollider>
{
    static constexpr int type = CAPSULECOLLIDER_TYPE;
};
template <> struct IsCollider<CapsuleCollider>
{
    static constexpr bool value = true;
};
template <> struct IsCapsuleCollider<CapsuleCollider>
{
    static constexpr bool value = true;
};
template <> struct IsComponent<CapsuleCollider>
{
    static constexpr bool value = true;
};
template <> struct IsComponentInternal<CapsuleCollider>
{
    static constexpr bool value = true;
};
} // namespace PhysicsEngine

#endif