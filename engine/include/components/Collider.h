#ifndef COLLIDER_H__
#define COLLIDER_H__

#include "Component.h"

#include "../core/AABB.h"
#include "../core/Ray.h"

#define GLM_FORCE_RADIANS

#include "glm/glm.hpp"

namespace PhysicsEngine
{
class Collider : public Component
{
  public:
    Collider(World *world);
    Collider(World *world, Guid id);
    ~Collider();

    virtual void serialize(YAML::Node &out) const override;
    virtual void deserialize(const YAML::Node &in) override;

    virtual bool intersect(AABB aabb) const = 0;
};

template <> struct IsComponentInternal<Collider>
{
    static constexpr bool value = true;
};
} // namespace PhysicsEngine

#endif