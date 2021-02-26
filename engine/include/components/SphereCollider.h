#ifndef SPHERECOLLIDER_H__
#define SPHERECOLLIDER_H__

#include <vector>

#include "Collider.h"

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"

#include "../core/Sphere.h"

namespace PhysicsEngine
{

class SphereCollider : public Collider
{
  public:
    Sphere mSphere;

  public:
    SphereCollider();
    SphereCollider(Guid id);
    ~SphereCollider();

    virtual void serialize(std::ostream &out) const override;
    virtual void deserialize(std::istream &in) override;

    bool intersect(AABB aabb) const override;

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