#ifndef __BOXCOLLIDER_H__
#define __BOXCOLLIDER_H__

#include <vector>

#include "Collider.h"

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"

namespace PhysicsEngine
{
class BoxCollider : public Collider
{
  public:
    AABB mAABB;

  public:
    BoxCollider();
    BoxCollider(Guid id);
    ~BoxCollider();

    virtual void serialize(std::ostream &out) const;
    virtual void deserialize(std::istream &in);

    bool intersect(AABB aabb) const;

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