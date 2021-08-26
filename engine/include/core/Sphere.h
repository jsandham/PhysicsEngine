#ifndef SPHERE_H__
#define SPHERE_H__

#define GLM_FORCE_RADIANS
#include "GLM.h"
#include "glm/glm.hpp"
#include "yaml-cpp/yaml.h"

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
    ~Sphere();

    float getVolume() const;
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