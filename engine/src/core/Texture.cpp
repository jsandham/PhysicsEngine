#include <iostream>

#include "../../include/core/Texture.h"

using namespace PhysicsEngine;

Texture::Texture()
{

}

int Texture::getNumChannels() const
{
	return numChannels;
}

TextureDimension Texture::getDimension() const
{
	return dimension;
}

int Texture::calcNumChannels(TextureFormat format)
{
	int nChannels = 0;

	switch (format)
	{
	case Depth:
		nChannels = 1;
		break;
	case RG:
		nChannels = 2;
		break;
	case RGB:
		nChannels = 3;
		break;
	case RGBA:
		nChannels = 4;
		break;
	default:
		std::cout <<"Error: Texture: Invalid texture format" << std::endl;
	}

	return nChannels;
}