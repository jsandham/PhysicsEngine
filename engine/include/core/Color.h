#ifndef COLOR_H__
#define COLOR_H__

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"
#include "GLM.h"
#include "yaml-cpp/yaml.h"

namespace PhysicsEngine
{
class Color
{
  public:
    static const Color clear;
    static const Color white;
    static const Color black;
    static const Color red;
    static const Color green;
    static const Color blue;
    static const Color yellow;
    static const Color gray;
    static const Color cyan;
    static const Color magenta;

    float r;
    float g;
    float b;
    float a;

  public:
    Color();
    Color(float r, float g, float b, float a);
    Color(glm::vec4 rgba);
    ~Color();
};

class Color32
{
  public:
    static const Color32 clear;
    static const Color32 white;
    static const Color32 black;
    static const Color32 red;
    static const Color32 green;
    static const Color32 blue;
    static const Color32 yellow;
    static const Color32 gray;
    static const Color32 cyan;
    static const Color32 magenta;

    unsigned char r;
    unsigned char g;
    unsigned char b;
    unsigned char a;

  public:
    Color32();
    Color32(unsigned char r, unsigned char g, unsigned char b, unsigned char a);
    ~Color32();
};
} // namespace PhysicsEngine

namespace YAML
{
// Color
template <> struct convert<PhysicsEngine::Color>
{
    static Node encode(const PhysicsEngine::Color &rhs)
    {
        Node node;
        node.push_back(rhs.r);
        node.push_back(rhs.g);
        node.push_back(rhs.b);
        node.push_back(rhs.a);
        return node;
    }

    static bool decode(const Node &node, PhysicsEngine::Color &rhs)
    {
        if (!node.IsSequence() || node.size() != 4)
        {
            return false;
        }

        rhs.r = node[0].as<float>();
        rhs.g = node[1].as<float>();
        rhs.b = node[2].as<float>();
        rhs.a = node[3].as<float>();
        return true;
    }
};

// Color32
template <> struct convert<PhysicsEngine::Color32>
{
    static Node encode(const PhysicsEngine::Color32 &rhs)
    {
        Node node;
        node.push_back(rhs.r);
        node.push_back(rhs.g);
        node.push_back(rhs.b);
        node.push_back(rhs.a);
        return node;
    }

    static bool decode(const Node &node, PhysicsEngine::Color32 &rhs)
    {
        if (!node.IsSequence() || node.size() != 4)
        {
            return false;
        }

        rhs.r = node[0].as<unsigned char>();
        rhs.g = node[1].as<unsigned char>();
        rhs.b = node[2].as<unsigned char>();
        rhs.a = node[3].as<unsigned char>();
        return true;
    }
};
} // namespace YAML

#endif