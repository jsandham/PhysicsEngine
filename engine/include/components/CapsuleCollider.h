#ifndef CAPSULECOLLIDER_H__
#define CAPSULECOLLIDER_H__

#include <vector>

#include "Collider.h"

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"

#include "../core/Capsule.h"

namespace PhysicsEngine
{

class CapsuleCollider : public Collider
{
  public:
    Capsule mCapsule;

  public:
    CapsuleCollider();
    CapsuleCollider(Guid id);
    ~CapsuleCollider();

    virtual void serialize(YAML::Node &out) const override;
    virtual void deserialize(const YAML::Node &in) override;

    virtual int getType() const override;
    virtual std::string getObjectName() const override;

    bool intersect(AABB aabb) const override;
};

template <> struct ComponentType<CapsuleCollider>
{
    static constexpr int type = CAPSULECOLLIDER_TYPE;
};

template <> struct IsComponentInternal<CapsuleCollider>
{
    static constexpr bool value = true;
};
} // namespace PhysicsEngine

#endif