#ifndef SPHERE_H__
#define SPHERE_H__

#include "GlmYaml.h"

namespace PhysicsEngine
{
class Sphere
{
  public:
    glm::vec3 mCentre;
    float mRadius;

  public:
    Sphere();
    Sphere(const glm::vec3 &centre, float radius);

    float getVolume() const;
    glm::vec3 getNormal(const glm::vec3 &point) const;
};
} // namespace PhysicsEngine

namespace YAML
{
// Sphere
template <> struct convert<PhysicsEngine::Sphere>
{
    static Node encode(const PhysicsEngine::Sphere &rhs)
    {
        Node node;
        node["centre"] = rhs.mCentre;
        node["radius"] = rhs.mRadius;
        return node;
    }

    static bool decode(const Node &node, PhysicsEngine::Sphere &rhs)
    {
        rhs.mCentre = node["centre"].as<glm::vec3>();
        rhs.mRadius = node["radius"].as<float>();
        return true;
    }
};
} // namespace YAML

#endif