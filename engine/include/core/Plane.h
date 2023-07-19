#ifndef PLANE_H__
#define PLANE_H__

#define GLM_FORCE_RADIANS

#include "GLM.h"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "yaml-cpp/yaml.h"

namespace PhysicsEngine
{
// plane defined by n.x*x + n.y*y + n.z*z + d = 0, where d = -dot(n, x0)
class Plane
{
  public:
    glm::vec3 mNormal;
    glm::vec3 mX0;

  public:
    Plane();
    Plane(glm::vec3 normal, glm::vec3 x0);
    ~Plane();

    float signedDistance(const glm::vec3 &point) const;
};
} // namespace PhysicsEngine

namespace YAML
{
// Plane
template <> struct convert<PhysicsEngine::Plane>
{
    static Node encode(const PhysicsEngine::Plane &rhs)
    {
        Node node;
        node["normal"] = rhs.mNormal;
        node["x0"] = rhs.mX0;
        return node;
    }

    static bool decode(const Node &node, PhysicsEngine::Plane &rhs)
    {
        rhs.mNormal = node["normal"].as<glm::vec3>();
        rhs.mX0 = node["x0"].as<glm::vec3>();

        return true;
    }
};
} // namespace YAML

#endif