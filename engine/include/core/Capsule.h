#ifndef CAPSULE_H__
#define CAPSULE_H__

#include "GlmYaml.h"

namespace PhysicsEngine
{
class Capsule
{
  public:
    glm::vec3 mCentre;
    float mRadius;
    float mHeight;

  public:
    Capsule();
    Capsule(glm::vec3 centre, float radius, float height);
    ~Capsule();
};
} // namespace PhysicsEngine

namespace YAML
{
// Capsule
template <> struct convert<PhysicsEngine::Capsule>
{
    static Node encode(const PhysicsEngine::Capsule &rhs)
    {
        Node node;
        node["centre"] = rhs.mCentre;
        node["radius"] = rhs.mRadius;
        node["height"] = rhs.mHeight;
        return node;
    }

    static bool decode(const Node &node, PhysicsEngine::Capsule &rhs)
    {
        rhs.mCentre = node["centre"].as<glm::vec3>();
        rhs.mRadius = node["radius"].as<float>();
        rhs.mHeight = node["height"].as<float>();
        return true;
    }
};
} // namespace YAML

#endif