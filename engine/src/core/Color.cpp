#include "../../include/core/Color.h"

using namespace PhysicsEngine;

const Color Color::clear(0.0f, 0.0f, 0.0f, 0.0f);
const Color Color::white(1.0f, 1.0f, 1.0f, 1.0f);
const Color Color::black(0, 0, 0, 1.0f);
const Color Color::red(1.0f, 0, 0, 1.0f);
const Color Color::green(0, 1.0f, 0, 1.0f);
const Color Color::blue(0, 0, 1.0f, 1.0f);
const Color Color::yellow(1.0f, 0.91764705f, 0.01568627f, 1.0f);
const Color Color::gray(0.5f, 0.5f, 0.5f, 1.0f);
const Color Color::cyan(0.0f, 1.0f, 1.0f, 1.0f);
const Color Color::magenta(1.0f, 0.0f, 1.0f, 1.0f);

Color::Color() : mR(0.0f), mG(0.0f), mB(0.0f), mA(0.0f)
{
}

Color::Color(float r, float g, float b, float a) : mR(r), mG(g), mB(b), mA(a)
{
}

Color::Color(const glm::vec4 &rgba) : mR(rgba.x), mG(rgba.y), mB(rgba.z), mA(rgba.w)
{
}

Color::~Color()
{
}

const Color32 Color32::clear(0, 0, 0, 0);
const Color32 Color32::white(255, 255, 255, 255);
const Color32 Color32::black(0, 0, 0, 255);
const Color32 Color32::red(255, 0, 0, 255);
const Color32 Color32::green(0, 255, 0, 255);
const Color32 Color32::blue(0, 0, 255, 255);
const Color32 Color32::yellow(255, 234, 4, 255);
const Color32 Color32::gray(127, 127, 127, 255);
const Color32 Color32::cyan(0, 255, 255, 255);
const Color32 Color32::magenta(255, 0, 255, 255);

Color32::Color32() : mR(0), mG(0), mB(0), mA(0)
{
}

Color32::Color32(unsigned char r, unsigned char g, unsigned char b, unsigned char a) : mR(r), mG(g), mB(b), mA(a)
{
}

Color32::~Color32()
{
}