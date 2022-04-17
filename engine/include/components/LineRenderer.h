#ifndef LINERENDERER_H__
#define LINERENDERER_H__

#include "Component.h"

#define GLM_FORCE_RADIANS

#include "glm/glm.hpp"

namespace PhysicsEngine
{
class LineRenderer : public Component
{
  public:
    glm::vec3 mStart;
    glm::vec3 mEnd;
    bool mEnabled;

    Id mMaterialId;

  public:
    LineRenderer(World *world);
    LineRenderer(World *world, Id id);
    ~LineRenderer();

    virtual void serialize(YAML::Node &out) const override;
    virtual void deserialize(const YAML::Node &in) override;

    virtual int getType() const override;
    virtual std::string getObjectName() const override;
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