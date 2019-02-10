#include <iostream>

#include "../../include/core/PoolAllocator.h"
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

Texture3D::Texture3D(std::vector<char> data)
{
	size_t index = sizeof(int);
	Texture3DHeader* header = reinterpret_cast<Texture3DHeader*>(&data[index]);

	assetId = header->textureId;
	width = header->width;
	height = header->height;
	depth = header->depth;
	numChannels = header->numChannels;
	dimension = static_cast<TextureDimension>(header->dimension);
	format = static_cast<TextureFormat>(header->format);
	
	index += sizeof(Texture3DHeader);

	rawTextureData.resize(header->textureSize);
	for(size_t i = 0; i < header->textureSize; i++){
		rawTextureData[i] = *reinterpret_cast<unsigned char*>(&data[index + sizeof(unsigned char) * i]);
	}

	index += rawTextureData.size() * sizeof(unsigned char);

	std::cout << "Texture3D index: " << index << " data size: " << data.size() << std::endl;
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

void* Texture3D::operator new(size_t size)
{
	return getAllocator<Texture3D>().allocate();
}

void Texture3D::operator delete(void*)
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
