#ifndef AABB_H__
#define AABB_H__

#define GLM_FORCE_RADIANS
#include "glm/glm.hpp"
#include "GLM.h"
#include "yaml-cpp/yaml.h"

namespace PhysicsEngine
{
class AABB
{
  public:
    glm::vec3 mCentre;
    glm::vec3 mSize;

  public:
    AABB();
    AABB(glm::vec3 centre, glm::vec3 size);
    ~AABB();

    glm::vec3 getExtents() const;
    glm::vec3 getMin() const;
    glm::vec3 getMax() const;
};
} // namespace PhysicsEngine

namespace YAML
{
// AABB
template <> struct convert<PhysicsEngine::AABB>
{
    static Node encode(const PhysicsEngine::AABB &rhs)
    {
        Node node;
        node["centre"] = rhs.mCentre;
        node["size"] = rhs.mSize;
        return node;
    }

    static bool decode(const Node &node, PhysicsEngine::AABB &rhs)
    {
        rhs.mCentre = node["centre"].as<glm::vec3>();
        rhs.mSize = node["size"].as<glm::vec3>();
        return true;
    }
};
} // namespace YAML

#endif