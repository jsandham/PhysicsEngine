#ifndef __COLOR_H__
#define __COLOR_H__

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"

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

#endif