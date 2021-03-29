#ifndef VIEWPORT_H__
#define VIEWPORT_H__

#include "GLM.h"
#include "yaml-cpp/yaml.h"

namespace PhysicsEngine
{
class Viewport
{
  public:
    int mX;
    int mY;
    int mWidth;
    int mHeight;

  public:
    Viewport();
    Viewport(int x, int y, int width, int height);
    ~Viewport();
};
} // namespace PhysicsEngine

namespace YAML
{
// Viewport
template <> struct convert<PhysicsEngine::Viewport>
{
    static Node encode(const PhysicsEngine::Viewport &rhs)
    {
        Node node;
        node["x"] = rhs.mX;
        node["y"] = rhs.mY;
        node["width"] = rhs.mWidth;
        node["height"] = rhs.mHeight;
        return node;
    }

    static bool decode(const Node &node, PhysicsEngine::Viewport &rhs)
    {
        rhs.mX = node["x"].as<int>();
        rhs.mY = node["y"].as<int>();
        rhs.mWidth = node["width"].as<int>();
        rhs.mHeight = node["height"].as<int>();
        return true;
    }
};
} // namespace YAML

#endif