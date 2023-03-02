#ifndef COLOR_H__
#define COLOR_H__

#define GLM_FORCE_RADIANS

#include "GLM.h"
#include "glm/glm.hpp"
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

    float mR;
    float mG;
    float mB;
    float mA;

  public:
    Color();
    Color(float r, float g, float b, float a);
    Color(const glm::vec4& rgba);
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

    unsigned char mR;
    unsigned char mG;
    unsigned char mB;
    unsigned char mA;

  public:
    Color32();
    Color32(unsigned char r, unsigned char g, unsigned char b, unsigned char a);
    ~Color32();

    // Allows use of Color32 as key in unordered_map
    bool operator==(const Color32& other) const
    {
        return ((mR == other.mR) && (mG == other.mG) && (mB == other.mB) && (mA == other.mA));
    }

    static uint32_t convertColor32ToUint32(const Color32 &c);
    static Color32 convertUint32ToColor32(uint32_t i);
};
} // namespace PhysicsEngine

namespace std 
{
    // Allows use of Color32 as key in unordered_map
    template <>
    struct hash<PhysicsEngine::Color32>
    {
        std::size_t operator()(const PhysicsEngine::Color32& color) const
        {
            using std::size_t;
            using std::hash;
            using std::string;

            size_t r = color.mR;
            size_t g = color.mG;
            size_t b = color.mB;
            size_t a = color.mA;

            return r + 255 * g + 255 * 255 * b + 255 * 255 * 255 * a;
        }
    };
}

namespace YAML
{
// Color
template <> struct convert<PhysicsEngine::Color>
{
    static Node encode(const PhysicsEngine::Color &rhs)
    {
        Node node;
        node.push_back(rhs.mR);
        node.push_back(rhs.mG);
        node.push_back(rhs.mB);
        node.push_back(rhs.mA);
        return node;
    }

    static bool decode(const Node &node, PhysicsEngine::Color &rhs)
    {
        if (!node.IsSequence() || node.size() != 4)
        {
            return false;
        }

        rhs.mR = node[0].as<float>();
        rhs.mG = node[1].as<float>();
        rhs.mB = node[2].as<float>();
        rhs.mA = node[3].as<float>();
        return true;
    }
};

// Color32
template <> struct convert<PhysicsEngine::Color32>
{
    static Node encode(const PhysicsEngine::Color32 &rhs)
    {
        Node node;
        node.push_back(rhs.mR);
        node.push_back(rhs.mG);
        node.push_back(rhs.mB);
        node.push_back(rhs.mA);
        return node;
    }

    static bool decode(const Node &node, PhysicsEngine::Color32 &rhs)
    {
        if (!node.IsSequence() || node.size() != 4)
        {
            return false;
        }

        rhs.mR = node[0].as<unsigned char>();
        rhs.mG = node[1].as<unsigned char>();
        rhs.mB = node[2].as<unsigned char>();
        rhs.mA = node[3].as<unsigned char>();
        return true;
    }
};
} // namespace YAML

#endif