#ifndef __COLLIDER_H__
#define __COLLIDER_H__

#include "Component.h"

#include "../core/AABB.h"
#include "../core/Ray.h"

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"

namespace PhysicsEngine
{
class Collider : public Component
{
  public:
    Collider();
    virtual ~Collider() = 0;

    virtual bool intersect(AABB aabb) const = 0;
};

template <typename T> struct IsCollider
{
    static constexpr bool value = false;
};

template <> struct IsCollider<Collider>
{
    static constexpr bool value = true;
};
template <> struct IsComponent<Collider>
{
    static constexpr bool value = true;
};
template <> struct IsComponentInternal<Collider>
{
    static constexpr bool value = true;
};
} // namespace PhysicsEngine

#endif