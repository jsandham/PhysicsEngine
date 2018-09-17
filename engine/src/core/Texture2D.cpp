#include <iostream>

#include "../../include/core/Texture2D.h"
#include "../../include/core/Log.h"
#include "../../include/graphics/Graphics.h"

using namespace PhysicsEngine;

Texture2D::Texture2D()
{
	this->dimension = TextureDimension::Tex2D;

	this->width = 0;
	this->height = 0;
	this->format = TextureFormat::RGB;

	this->numChannels = calcNumChannels(format);
}

Texture2D::Texture2D(int width, int height)
{
	this->dimension = TextureDimension::Tex2D;

	this->width = width;
	this->height = height;
	this->format = TextureFormat::RGB;

	this->numChannels = calcNumChannels(format);

	rawTextureData.resize(width*height*numChannels);
}

Texture2D::Texture2D(int width, int height, TextureFormat format)
{
	this->dimension = TextureDimension::Tex2D;

	this->width = width;
	this->height = height;
	this->format = format;

	this->numChannels = calcNumChannels(format);

	rawTextureData.resize(width*height*numChannels);
}

Texture2D::~Texture2D()
{
	
}

std::vector<unsigned char> Texture2D::getRawTextureData()
{
	return rawTextureData;
}

std::vector<Color> Texture2D::getPixels()
{
	std::vector<Color> colors;

	colors.resize(width*height);

	for (unsigned int i = 0; i < colors.size(); i++){
		colors[i].r = rawTextureData[numChannels * i];
		colors[i].g = rawTextureData[numChannels * i + 1];
		colors[i].b = rawTextureData[numChannels * i + 2];
	}

	return colors;
}

Color Texture2D::getPixel(int x, int y)
{
	int index = numChannels * (x + width * y);

	if (index + numChannels >= rawTextureData.size()){
		Log::Error("Texture2D: pixel index out of range");
	}

	Color color;
	if (numChannels == 1){
		color.r = rawTextureData[index];
		color.g = rawTextureData[index];
		color.b = rawTextureData[index];
		color.a = 0;
	}
	else if (numChannels == 3){
		color.r = rawTextureData[index];
		color.g = rawTextureData[index + 1];
		color.b = rawTextureData[index + 2];
		color.a = 0;
	}
	else if (numChannels == 4){
		color.r = rawTextureData[index];
		color.g = rawTextureData[index + 1];
		color.b = rawTextureData[index + 2];
		color.a = rawTextureData[index + 3];
	}

	return color;
}

TextureFormat Texture2D::getFormat()
{
	return format;
}

void Texture2D::setRawTextureData(std::vector<unsigned char> data, int width, int height, TextureFormat format)
{
	switch(format)
	{
		case TextureFormat::Depth:
			numChannels = 1;
			break;
		case TextureFormat::RG:
			numChannels = 2;
			break;
		case TextureFormat::RGB:
			numChannels = 3;
			break;
		case TextureFormat::RGBA:
			numChannels = 4;
			break;
		default:
			std::cout << "Error: Unsupported texture format " << format << std::endl;
			return;
	}

	this->width = width;
	this->height = height;
	this->format = format;

	rawTextureData = data;
}

void Texture2D::setPixels(std::vector<Color> colors)
{
	if (colors.size() != width*height){
		Log::Error("Texture2D: error when trying to set pixels. Number of colors must match dimensions of texture");
		return;
	}

	for (unsigned int i = 0; i < colors.size(); i++){
		if (numChannels == 1){
			rawTextureData[i] = colors[i].r;
		}
		else if (numChannels == 3){
			rawTextureData[i] = colors[i].r;
			rawTextureData[i + 1] = colors[i].g;
			rawTextureData[i + 2] = colors[i].b;
		}
		else if (numChannels == 4){
			rawTextureData[i] = colors[i].r;
			rawTextureData[i + 1] = colors[i].g;
			rawTextureData[i + 2] = colors[i].b;
			rawTextureData[i + 3] = colors[i].a;
		}
	}
}

void Texture2D::setPixel(int x, int y, Color color)
{
	int index = numChannels * (x + width * y);

	if (index + numChannels >= rawTextureData.size()){
		Log::Error("Texture2D: pixel index out of range");
		return;
	}

	if (numChannels == 1){
		rawTextureData[index] = color.r;
	}
	else if (numChannels == 3){
		rawTextureData[index] = color.r;
		rawTextureData[index + 1] = color.g;
		rawTextureData[index + 2] = color.b;
	}
	else if (numChannels == 4){
		rawTextureData[index] = color.r;
		rawTextureData[index + 1] = color.g;
		rawTextureData[index + 2] = color.b;
		rawTextureData[index + 3] = color.a;
	}
}

void Texture2D::readPixels()
{
	Graphics::readPixels(this);
}

void Texture2D::apply()
{
	Graphics::apply(this);
}






// #include <iostream>
// #include "OpenGL.h"
// #include "Texture2D.h"

// #include "../core/Log.h"

// #include "../stb_image/stb_image.h"

// using namespace PhysicsEngine;

// Texture2D::Texture2D()
// {
// 	this->dimension = TextureDimension::Tex2D;

// 	this->width = 0;
// 	this->height = 0;
// 	this->format = TextureFormat::RGB;

// 	this->numChannels = calcNumChannels(format);
// }

// Texture2D::Texture2D(int width, int height)
// {
// 	this->dimension = TextureDimension::Tex2D;

// 	this->width = width;
// 	this->height = height;
// 	this->format = TextureFormat::RGB;

// 	this->numChannels = calcNumChannels(format);

// 	rawTextureData.resize(width*height*numChannels);
// }

// Texture2D::Texture2D(int width, int height, TextureFormat format)
// {
// 	this->dimension = TextureDimension::Tex2D;

// 	this->width = width;
// 	this->height = height;
// 	this->format = format;

// 	this->numChannels = calcNumChannels(format);

// 	rawTextureData.resize(width*height*numChannels);
// }

// Texture2D::~Texture2D()
// {
	
// }

// void Texture2D::generate()
// {
// 	glGenTextures(1, &handle);
// 	glBindTexture(GL_TEXTURE_2D, handle);

// 	GLenum openglFormat = OpenGL::getTextureFormat(format);

// 	glTexImage2D(GL_TEXTURE_2D, 0, openglFormat, width, height, 0, openglFormat, GL_UNSIGNED_BYTE, &rawTextureData[0]);

// 	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
// 	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
// 	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
// 	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

// 	glBindTexture(GL_TEXTURE_2D, 0);
// }

// void Texture2D::destroy()
// {
// 	glDeleteTextures(1, &handle);
// }

// void Texture2D::bind()
// {
// 	glBindTexture(GL_TEXTURE_2D, handle);
// }

// void Texture2D::unbind()
// {
// 	glBindTexture(GL_TEXTURE_2D, 0);
// }

// void Texture2D::active(unsigned int slot){
// 	glActiveTexture(GL_TEXTURE0 + slot);
// }

// std::vector<unsigned char> Texture2D::getRawTextureData()
// {
// 	return rawTextureData;
// }

// std::vector<Color> Texture2D::getPixels()
// {
// 	std::vector<Color> colors;

// 	colors.resize(width*height);

// 	for (unsigned int i = 0; i < colors.size(); i++){
// 		colors[i].r = rawTextureData[numChannels * i];
// 		colors[i].g = rawTextureData[numChannels * i + 1];
// 		colors[i].b = rawTextureData[numChannels * i + 2];
// 	}

// 	return colors;
// }

// Color Texture2D::getPixel(int x, int y)
// {
// 	int index = numChannels * (x + width * y);

// 	if (index + numChannels >= rawTextureData.size()){
// 		Log::Error("Texture2D: pixel index out of range");
// 	}

// 	Color color;
// 	if (numChannels == 1){
// 		color.r = rawTextureData[index];
// 		color.g = rawTextureData[index];
// 		color.b = rawTextureData[index];
// 		color.a = 0;
// 	}
// 	else if (numChannels == 3){
// 		color.r = rawTextureData[index];
// 		color.g = rawTextureData[index + 1];
// 		color.b = rawTextureData[index + 2];
// 		color.a = 0;
// 	}
// 	else if (numChannels == 4){
// 		color.r = rawTextureData[index];
// 		color.g = rawTextureData[index + 1];
// 		color.b = rawTextureData[index + 2];
// 		color.a = rawTextureData[index + 3];
// 	}

// 	return color;
// }

// void Texture2D::setRawTextureData(std::vector<unsigned char> data)
// {
// 	if (width*height*numChannels != data.size()){
// 		Log::Error("Texture2D: raw texture data does not match size of texture");
// 		return;
// 	}

// 	rawTextureData = data;
// }

// void Texture2D::setPixels(std::vector<Color> colors)
// {
// 	if (colors.size() != width*height){
// 		Log::Error("Texture2D: error when trying to set pixels. Number of colors must match dimensions of texture");
// 		return;
// 	}

// 	for (unsigned int i = 0; i < colors.size(); i++){
// 		if (numChannels == 1){
// 			rawTextureData[i] = colors[i].r;
// 		}
// 		else if (numChannels == 3){
// 			rawTextureData[i] = colors[i].r;
// 			rawTextureData[i + 1] = colors[i].g;
// 			rawTextureData[i + 2] = colors[i].b;
// 		}
// 		else if (numChannels == 4){
// 			rawTextureData[i] = colors[i].r;
// 			rawTextureData[i + 1] = colors[i].g;
// 			rawTextureData[i + 2] = colors[i].b;
// 			rawTextureData[i + 3] = colors[i].a;
// 		}
// 	}
// }

// void Texture2D::setPixel(int x, int y, Color color)
// {
// 	int index = numChannels * (x + width * y);

// 	if (index + numChannels >= rawTextureData.size()){
// 		Log::Error("Texture2D: pixel index out of range");
// 		return;
// 	}

// 	if (numChannels == 1){
// 		rawTextureData[index] = color.r;
// 	}
// 	else if (numChannels == 3){
// 		rawTextureData[index] = color.r;
// 		rawTextureData[index + 1] = color.g;
// 		rawTextureData[index + 2] = color.b;
// 	}
// 	else if (numChannels == 4){
// 		rawTextureData[index] = color.r;
// 		rawTextureData[index + 1] = color.g;
// 		rawTextureData[index + 2] = color.b;
// 		rawTextureData[index + 3] = color.a;
// 	}
// }

// void Texture2D::readPixels()
// {
// 	glBindTexture(GL_TEXTURE_2D, handle);

// 	GLenum openglFormat = OpenGL::getTextureFormat(format);

// 	glGetTextureImage(handle, 0, openglFormat, GL_UNSIGNED_BYTE, width*height*numChannels, &rawTextureData[0]);
	
// 	glBindTexture(GL_TEXTURE_2D, 0);
// }

// void Texture2D::apply()
// {
// 	glBindTexture(GL_TEXTURE_2D, handle);

// 	GLenum openglFormat = OpenGL::getTextureFormat(format);

// 	glTexImage2D(GL_TEXTURE_2D, 0, openglFormat, width, height, 0, openglFormat, GL_UNSIGNED_BYTE, &rawTextureData[0]);

// 	glBindTexture(GL_TEXTURE_2D, 0);
// }

// GLuint Texture2D::getHandle() const
// {
// 	return handle;
// }