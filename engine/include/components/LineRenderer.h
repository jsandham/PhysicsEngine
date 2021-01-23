#ifndef __LINERENDERER_H__
#define __LINERENDERER_H__

#include <vector>

#include "Component.h"

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"

namespace PhysicsEngine
{
#pragma pack(push, 1)
struct LineRendererHeader
{
    Guid mComponentId;
    Guid mEntityId;
    Guid mMaterialId;
    glm::vec3 mStart;
    glm::vec3 mEnd;
};
#pragma pack(pop)

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

    std::vector<char> serialize() const;
    std::vector<char> serialize(const Guid &componentId, const Guid &entityId) const;
    void deserialize(const std::vector<char> &data);
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