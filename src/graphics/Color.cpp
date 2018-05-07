#include "Color.h"

using namespace PhysicsEngine;

Color Color::white(255, 255, 255, 255);
Color Color::black(0, 0, 0, 255);
Color Color::red(255, 0, 0, 255);
Color Color::green(0, 255, 0, 255);
Color Color::blue(0, 0, 255, 255);
Color Color::yellow(255, 234, 4, 255);

Color::Color()
{
	r = 0;
	g = 0;
	b = 0;
	a = 0;
}

Color::Color(unsigned char r, unsigned char g, unsigned char b, unsigned char a)
{
	this->r = r;
	this->g = g;
	this->b = b;
	this->a = a;
}


Color::~Color()
{

}