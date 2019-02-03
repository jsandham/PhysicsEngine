#include "../../include/core/Texture3D.h"
#include "../../include/core/Log.h"
#include "../../include/graphics/Graphics.h"

using namespace PhysicsEngine;

Texture3D::Texture3D()
{
	this->dimension = TextureDimension::Tex2D;

	this->width = 0;
	this->height = 0;
	this->depth = 0;
	this->format = TextureFormat::RGB;

	this->numChannels = calcNumChannels(format);
}

Texture3D::Texture3D(unsigned char* data)
{
	
}

Texture3D::Texture3D(int width, int height, int depth, int numChannels)
{
	this->dimension = TextureDimension::Tex2D;

	this->width = width;
	this->height = height;
	this->depth = depth;
	this->format = TextureFormat::RGB;

	this->numChannels = calcNumChannels(format);
}

Texture3D::~Texture3D()
{

}

int Texture3D::getWidth() const
{
	return width;
}

int Texture3D::getHeight() const
{
	return height;
}

int Texture3D::getDepth() const
{
	return depth;
}

void Texture3D::redefine(int width, int height, int depth, TextureFormat format)
{
	this->width = width;
	this->height = height;
	this->depth = depth;
	this->format = TextureFormat::RGB;

	this->numChannels = calcNumChannels(format);
}

std::vector<unsigned char> Texture3D::getRawTextureData()
{
	return rawTextureData;
}

Color Texture3D::getPixel(int x, int y, int z)
{
	return Color::white;
}

TextureFormat Texture3D::getFormat()
{
	return format;
}

void Texture3D::setRawTextureData(std::vector<unsigned char> data)
{

}

void Texture3D::setPixel(int x, int y, int z, Color color)
{

}

void Texture3D::readPixels()
{
	Graphics::readPixels(this);
}

void Texture3D::apply()
{
	Graphics::apply(this);
}
