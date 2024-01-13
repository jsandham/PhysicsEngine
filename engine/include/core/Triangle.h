#ifndef TRIANGLE_H__
#define TRIANGLE_H__

#include "GlmYaml.h"

namespace PhysicsEngine
{
class Triangle
{
  public:
    glm::vec3 mV0;
    glm::vec3 mV1;
    glm::vec3 mV2;

  public:
    Triangle();
    Triangle(const glm::vec3 &v0, const glm::vec3 &v1, const glm::vec3 &v2);
    ~Triangle();

    glm::vec3 getBarycentric(const glm::vec3 &p);
};
} // namespace PhysicsEngine

namespace YAML
{
// Triangle
template <> struct convert<PhysicsEngine::Triangle>
{
    static Node encode(const PhysicsEngine::Triangle &rhs)
    {
        Node node;
        node["v0"] = rhs.mV0;
        node["v1"] = rhs.mV1;
        node["v2"] = rhs.mV2;
        return node;
    }

    static bool decode(const Node &node, PhysicsEngine::Triangle &rhs)
    {
        rhs.mV0 = node["v0"].as<glm::vec3>();
        rhs.mV1 = node["v1"].as<glm::vec3>();
        rhs.mV2 = node["v2"].as<glm::vec3>();

        return true;
    }
};
} // namespace YAML

#endif