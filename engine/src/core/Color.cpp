#include "../../include/core/Color.h"

using namespace PhysicsEngine;

Color Color::white(1.0f, 1.0f, 1.0f, 1.0f);
Color Color::black(0, 0, 0, 1.0f);
Color Color::red(1.0f, 0, 0, 1.0f);
Color Color::green(0, 1.0f, 0, 1.0f);
Color Color::blue(0, 0, 1.0f, 1.0f);
Color Color::yellow(1.0f, 0.91764705f, 0.01568627f, 1.0f);

Color::Color()
{
	r = 0.0f;
	g = 0.0f;
	b = 0.0f;
	a = 0.0f;
}

Color::Color(float r, float g, float b, float a)
{
	this->r = r;
	this->g = g;
	this->b = b;
	this->a = a;
}


Color::~Color()
{

}

Color32 Color32::white(255, 255, 255, 255);
Color32 Color32::black(0, 0, 0, 255);
Color32 Color32::red(255, 0, 0, 255);
Color32 Color32::green(0, 255, 0, 255);
Color32 Color32::blue(0, 0, 255, 255);
Color32 Color32::yellow(255, 234, 4, 255);

Color32::Color32()
{
	r = 0;
	g = 0;
	b = 0;
	a = 0;
}

Color32::Color32(unsigned char r, unsigned char g, unsigned char b, unsigned char a)
{
	this->r = r;
	this->g = g;
	this->b = b;
	this->a = a;
}


Color32::~Color32()
{

}