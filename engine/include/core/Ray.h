#ifndef RAY_H__
#define RAY_H__

#include "GlmYaml.h"

namespace PhysicsEngine
{
class Ray
{
  public:
    glm::vec3 mOrigin;
    glm::vec3 mDirection;

  public:
    Ray();
    Ray(const glm::vec3 &origin, const glm::vec3 &direction);

    glm::vec3 getPoint(float t) const;
};
} // namespace PhysicsEngine

namespace YAML
{
// Ray
template <> struct convert<PhysicsEngine::Ray>
{
    static Node encode(const PhysicsEngine::Ray &rhs)
    {
        Node node;
        node["origin"] = rhs.mOrigin;
        node["direction"] = rhs.mDirection;
        return node;
    }

    static bool decode(const Node &node, PhysicsEngine::Ray &rhs)
    {
        rhs.mOrigin = node["origin"].as<glm::vec3>();
        rhs.mDirection = node["direction"].as<glm::vec3>();

        return true;
    }
};
} // namespace YAML

#endif