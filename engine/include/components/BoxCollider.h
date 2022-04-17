#ifndef BOXCOLLIDER_H__
#define BOXCOLLIDER_H__

#include "Collider.h"

#define GLM_FORCE_RADIANS

#include "glm/glm.hpp"

#include "../core/AABB.h"

namespace PhysicsEngine
{
class BoxCollider : public Collider
{
  public:
    AABB mAABB;

  public:
    BoxCollider(World *world);
    BoxCollider(World *world, Id id);
    ~BoxCollider();

    virtual void serialize(YAML::Node &out) const override;
    virtual void deserialize(const YAML::Node &in) override;

    virtual int getType() const override;
    virtual std::string getObjectName() const override;

    bool intersect(AABB aabb) const override;

    std::vector<float> getLines() const;
};

template <> struct ComponentType<BoxCollider>
{
    static constexpr int type = BOXCOLLIDER_TYPE;
};

template <> struct IsComponentInternal<BoxCollider>
{
    static constexpr bool value = true;
};
} // namespace PhysicsEngine

#endif