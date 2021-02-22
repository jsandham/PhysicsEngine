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
    Collider(Guid id);
    ~Collider();

    virtual void serialize(std::ostream &out) const;
    virtual void deserialize(std::istream &in);

    virtual bool intersect(AABB aabb) const = 0;
};

template <> struct IsComponentInternal<Collider>
{
    static constexpr bool value = true;
};
} // namespace PhysicsEngine

#endif