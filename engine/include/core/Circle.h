#ifndef CIRCLE_H__
#define CIRCLE_H__

#define GLM_FORCE_RADIANS
#include "GLM.h"
#include "glm/glm.hpp"
#include "glm/gtc/constants.hpp"
#include "yaml-cpp/yaml.h"

namespace PhysicsEngine
{
class Circle
{
  public:
    glm::vec3 mCentre;
    glm::vec3 mNormal;
    float mRadius;

  public:
    Circle();
    Circle(glm::vec3 centre, glm::vec3 normal, float radius);
    ~Circle();

    float getArea() const;
    float getCircumference() const;
};
} // namespace PhysicsEngine

namespace YAML
{
// Circle
template <> struct convert<PhysicsEngine::Circle>
{
    static Node encode(const PhysicsEngine::Circle &rhs)
    {
        Node node;
        node["centre"] = rhs.mCentre;
        node["normal"] = rhs.mNormal;
        node["radius"] = rhs.mRadius;
        return node;
    }

    static bool decode(const Node &node, PhysicsEngine::Circle &rhs)
    {
        rhs.mCentre = node["centre"].as<glm::vec3>();
        rhs.mNormal = node["normal"].as<glm::vec3>();
        rhs.mRadius = node["radius"].as<float>();

        return true;
    }
};
} // namespace YAML

#endif