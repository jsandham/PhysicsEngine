#ifndef LINE_H__
#define LINE_H__

#define GLM_FORCE_RADIANS

#include "GLM.h"
#include "glm/glm.hpp"
#include "yaml-cpp/yaml.h"

namespace PhysicsEngine
{
class Line
{
  public:
    glm::vec3 mStart;
    glm::vec3 mEnd;

  public:
    Line();
    Line(glm::vec3 start, glm::vec3 end);
    ~Line();

    float getLength() const;
};
} // namespace PhysicsEngine

namespace YAML
{
// Line
template <> struct convert<PhysicsEngine::Line>
{
    static Node encode(const PhysicsEngine::Line &rhs)
    {
        Node node;
        node["start"] = rhs.mStart;
        node["end"] = rhs.mEnd;
        return node;
    }

    static bool decode(const Node &node, PhysicsEngine::Line &rhs)
    {
        rhs.mStart = node["start"].as<glm::vec3>();
        rhs.mEnd = node["end"].as<glm::vec3>();

        return true;
    }
};
} // namespace YAML

#endif