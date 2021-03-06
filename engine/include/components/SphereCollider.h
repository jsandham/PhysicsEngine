#ifndef __SPHERECOLLIDER_H__
#define __SPHERECOLLIDER_H__

#include <vector>

#include "Collider.h"

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"

#include "../core/Sphere.h"

namespace PhysicsEngine
{
#pragma pack(push, 1)
struct SphereColliderHeader
{
    Guid mComponentId;
    Guid mEntityId;
    Sphere mSphere;
};
#pragma pack(pop)

class SphereCollider : public Collider
{
  public:
    Sphere mSphere;

  public:
    SphereCollider();
    SphereCollider(Guid id);
    ~SphereCollider();

    std::vector<char> serialize() const;
    std::vector<char> serialize(const Guid &componentId, const Guid &entityId) const;
    void deserialize(const std::vector<char> &data);

    void serialize(std::ostream& out) const;
    void deserialize(std::istream& in);

    bool intersect(AABB aabb) const;

    std::vector<float> getLines() const;
};

template <> struct ComponentType<SphereCollider>
{
    static constexpr int type = PhysicsEngine::SPHERECOLLIDER_TYPE;
};

template <> struct IsComponentInternal<SphereCollider>
{
    static constexpr bool value = true;
};
} // namespace PhysicsEngine

#endif