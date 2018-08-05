#include <iostream>
#include "Texture.h"

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
		std::cout <<"Texture: Invalid texture format" << std::endl;
	}

	return nChannels;
}






// #include <iostream>
// #include "Texture.h"

// using namespace PhysicsEngine;


// Texture::Texture()
// {
// }

// int Texture::getWidth() const
// {
// 	return width;
// }

// int Texture::getHeight() const
// {
// 	return height;
// }

// int Texture::getNumChannels() const
// {
// 	return numChannels;
// }

// TextureDimension Texture::getDimension() const
// {
// 	return dimension;
// }

// int Texture::calcNumChannels(TextureFormat format)
// {
// 	int nChannels = 0;

// 	switch (format)
// 	{
// 	case Red:
// 		nChannels = 1;
// 		break;
// 	case Green:
// 		nChannels = 1;
// 		break;
// 	case Blue:
// 		nChannels = 1;
// 		break;
// 	case Alpha:
// 		nChannels = 1;
// 		break;
// 	case Depth:
// 		nChannels = 1;
// 		break;
// 	case RGB:
// 		nChannels = 3;
// 		break;
// 	case BGR:
// 		nChannels = 3;
// 		break;
// 	case RGBA:
// 		nChannels = 4;
// 		break;
// 	case BGRA:
// 		nChannels = 4;
// 		break;
// 	default:
// 		std::cout <<"Texture: Invalid texture format" << std::endl;
// 	}

// 	return nChannels;
// }