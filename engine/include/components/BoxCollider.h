#ifndef __BOXCOLLIDER_H__
#define __BOXCOLLIDER_H__

#include <vector>

#include "Collider.h"

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"

namespace PhysicsEngine
{
#pragma pack(push, 1)
struct BoxColliderHeader
{
    Guid mComponentId;
    Guid mEntityId;
    AABB mAABB;
};
#pragma pack(pop)

class BoxCollider : public Collider
{
  public:
    AABB mAABB;

  public:
    BoxCollider();
    BoxCollider(Guid id);
    ~BoxCollider();

    std::vector<char> serialize() const;
    std::vector<char> serialize(const Guid &componentId, const Guid &entityId) const;
    void deserialize(const std::vector<char> &data);

    bool intersect(AABB aabb) const;

    std::vector<float> getLines() const;
};

template <typename T> struct IsBoxCollider
{
    static constexpr bool value = false;
};

template <> struct ComponentType<BoxCollider>
{
    static constexpr int type = BOXCOLLIDER_TYPE;
};
template <> struct IsCollider<BoxCollider>
{
    static constexpr bool value = true;
};
template <> struct IsBoxCollider<BoxCollider>
{
    static constexpr bool value = true;
};
template <> struct IsComponent<BoxCollider>
{
    static constexpr bool value = true;
};
template <> struct IsComponentInternal<BoxCollider>
{
    static constexpr bool value = true;
};
} // namespace PhysicsEngine

#endif