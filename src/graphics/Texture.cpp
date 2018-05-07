#include <iostream>
#include "Texture.h"

#include "../core/Log.h"

#include "../stb_image/stb_image.h"

using namespace PhysicsEngine;


Texture::Texture()
{
}

int Texture::getWidth() const
{
	return width;
}

int Texture::getHeight() const
{
	return height;
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
	case Red:
		nChannels = 1;
		break;
	case Green:
		nChannels = 1;
		break;
	case Blue:
		nChannels = 1;
		break;
	case Alpha:
		nChannels = 1;
		break;
	case Depth:
		nChannels = 1;
		break;
	case RGB:
		nChannels = 3;
		break;
	case BGR:
		nChannels = 3;
		break;
	case RGBA:
		nChannels = 4;
		break;
	case BGRA:
		nChannels = 4;
		break;
	default:
		Log::Error("Texture: Invalid texture format");
	}

	return nChannels;
}