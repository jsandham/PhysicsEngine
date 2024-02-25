#ifndef COLOR_H__
#define COLOR_H__

#include <string>

#include "glm.h"

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
    static const Color pink;
    static const Color silver;

    float mR;
    float mG;
    float mB;
    float mA;

  public:
    Color();
    Color(float r, float g, float b, float a);
    Color(const glm::vec4 &rgba);
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
    static const Color32 pink;
    static const Color32 silver;

    unsigned char mR;
    unsigned char mG;
    unsigned char mB;
    unsigned char mA;

  public:
    Color32();
    Color32(unsigned char r, unsigned char g, unsigned char b, unsigned char a);

    // Allows use of Color32 as key in unordered_map
    bool operator==(const Color32 &other) const
    {
        return ((mR == other.mR) && (mG == other.mG) && (mB == other.mB) && (mA == other.mA));
    }

    static uint32_t convertColor32ToUint32(const Color32 &c);
    static Color32 convertUint32ToColor32(uint32_t i);
    static Color32 convertColorToColor32(const Color& color);
};
} // namespace PhysicsEngine

namespace std
{
// Allows use of Color32 as key in unordered_map
template <> struct hash<PhysicsEngine::Color32>
{
    std::size_t operator()(const PhysicsEngine::Color32 &color) const
    {
        using std::hash;
        using std::size_t;
        using std::string;

        size_t r = color.mR;
        size_t g = color.mG;
        size_t b = color.mB;
        size_t a = color.mA;

        return r + 255 * g + 255 * 255 * b + 255 * 255 * 255 * a;
    }
};
} // namespace std

#endif