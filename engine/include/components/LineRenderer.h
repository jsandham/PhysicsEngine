#ifndef LINERENDERER_H__
#define LINERENDERER_H__

#include <vector>

#include "Component.h"

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"

namespace PhysicsEngine
{
class LineRenderer : public Component
{
  public:
    glm::vec3 mStart;
    glm::vec3 mEnd;

    Guid mMaterialId;

  public:
    LineRenderer();
    LineRenderer(Guid id);
    ~LineRenderer();

    virtual void serialize(std::ostream &out) const override;
    virtual void deserialize(std::istream &in) override;
    virtual void serialize(YAML::Node& out) const override;
    virtual void deserialize(const YAML::Node& in) override;
};

template <> struct ComponentType<LineRenderer>
{
    static constexpr int type = PhysicsEngine::LINERENDERER_TYPE;
};

template <> struct IsComponentInternal<LineRenderer>
{
    static constexpr bool value = true;
};
} // namespace PhysicsEngine

#endif