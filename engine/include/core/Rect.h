#ifndef RECT_H__
#define RECT_H__

#include "glm.h"
#include "yaml-cpp/yaml.h"

namespace PhysicsEngine
{
class Rect
{
  public:
    float mX;
    float mY;
    float mWidth;
    float mHeight;

  public:
    Rect();
    Rect(float x, float y, float width, float height);
    Rect(const glm::vec2 &min, const glm::vec2 &max);
    ~Rect();

    bool contains(float x, float y) const;
    float getArea() const;
    glm::vec2 getCentre() const;
    glm::vec2 getMin() const;
    glm::vec2 getMax() const;
};
} // namespace PhysicsEngine

namespace YAML
{
// AABB
template <> struct convert<PhysicsEngine::Rect>
{
    static Node encode(const PhysicsEngine::Rect &rhs)
    {
        Node node;
        node["x"] = rhs.mX;
        node["y"] = rhs.mY;
        node["width"] = rhs.mWidth;
        node["height"] = rhs.mHeight;
        return node;
    }

    static bool decode(const Node &node, PhysicsEngine::Rect &rhs)
    {
        rhs.mX = node["x"].as<float>();
        rhs.mY = node["y"].as<float>();
        rhs.mWidth = node["width"].as<float>();
        rhs.mHeight = node["height"].as<float>();
        return true;
    }
};
} // namespace YAML

#endif