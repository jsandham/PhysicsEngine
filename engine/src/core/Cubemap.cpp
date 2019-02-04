#include <iostream>

#include "../../include/core/PoolAllocator.h"
#include "../../include/core/Cubemap.h"
#include "../../include/core/Log.h"
#include "../../include/graphics/Graphics.h"

using namespace PhysicsEngine;

Cubemap::Cubemap()
{
	this->dimension = TextureDimension::Cube;
}

Cubemap::Cubemap(unsigned char* data)
{
	
}

Cubemap::Cubemap(int width)
{
	this->dimension = TextureDimension::Cube;

	this->width = width;
	this->format = TextureFormat::RGB;

	this->numChannels = calcNumChannels(format);

	rawTextureData.resize(6 * width*width*numChannels);
}

Cubemap::Cubemap(int width, TextureFormat format)
{
	this->dimension = TextureDimension::Cube;

	this->width = width;
	this->format = format;

	this->numChannels = calcNumChannels(format);

	rawTextureData.resize(6 * width*width*numChannels);
}

Cubemap::Cubemap(int width, int height, TextureFormat format)
{
	this->dimension = TextureDimension::Cube;

	this->width = width;
	this->format = format;

	this->numChannels = calcNumChannels(format);

	rawTextureData.resize(6 * width*width*numChannels);
}

Cubemap::~Cubemap()
{
	
}

void* Cubemap::operator new(size_t size)
{
	return getAllocator<Cubemap>().allocate();
}

void Cubemap::operator delete(void*)
{
	
}

int Cubemap::getWidth() const
{
	return width;
}

std::vector<unsigned char> Cubemap::getRawCubemapData()
{
	return rawTextureData;
}

std::vector<Color> Cubemap::getPixels(CubemapFace face)
{
	/*std::vector<Color> colors(width*width*numChannels);*/
	std::vector<Color> colors;

	int start = (int)face*width*width*numChannels;
	int end = start + width*width*numChannels;
	for (int i = start; i < end; i += numChannels){
		Color color;
		if (numChannels == 1){
			color.r = rawTextureData[i];
			color.g = rawTextureData[i];
			color.b = rawTextureData[i];
			color.a = 0;
		}
		else if (numChannels == 3){
			color.r = rawTextureData[i];
			color.g = rawTextureData[i + 1];
			color.b = rawTextureData[i + 2];
			color.a = 0;
		}
		else if (numChannels == 4){
			color.r = rawTextureData[i];
			color.g = rawTextureData[i + 1];
			color.b = rawTextureData[i + 2];
			color.a = rawTextureData[i + 3];
		}

		colors.push_back(color);
	}

	return colors;
}

Color Cubemap::getPixel(CubemapFace face, int x, int y)
{
	int index =  (int)face*width*width*numChannels + numChannels * (x + width * y);

	if (index + numChannels >= rawTextureData.size()){
		Log::Error("Cubemap: pixel index out of range");
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

void Cubemap::setRawCubemapData(std::vector<unsigned char> data)
{
	if (6*width*width*numChannels != data.size()){
		Log::Error("Cubemap: Raw texture data does not match size of cubemap");
		return;
	}

	rawTextureData = data;
}

void Cubemap::setPixels(CubemapFace face, int x, int y, Color color)
{

}

void Cubemap::setPixel(CubemapFace face, int x, int y, Color color)
{
	int index = (int)face*width*width*numChannels + numChannels * (x + width * y);

	if (index + numChannels >= rawTextureData.size()){
		Log::Error("Cubemap: pixel index out of range");
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

void Cubemap::readPixels()
{
	Graphics::readPixels(this);
}

void Cubemap::apply()
{
	Graphics::apply(this);
}























// #include "../../include/graphics/OpenGL.h"
// #include "../../include/graphics/Cubemap.h"

// #include <iostream>

// #include "../../include/core/Log.h"

// #include "../../include/stb_image/stb_image.h"

// using namespace PhysicsEngine;

// Cubemap::Cubemap()
// {
// 	this->dimension = TextureDimension::Cube;
// }

// Cubemap::Cubemap(int width)
// {
// 	this->dimension = TextureDimension::Cube;

// 	this->width = width;
// 	this->height = width;
// 	this->format = TextureFormat::RGB;

// 	this->numChannels = calcNumChannels(format);

// 	rawCubemapData.resize(6 * width*width*numChannels);
// }

// Cubemap::Cubemap(int width, TextureFormat format)
// {
// 	this->dimension = TextureDimension::Cube;

// 	this->width = width;
// 	this->height = width;
// 	this->format = format;

// 	this->numChannels = calcNumChannels(format);

// 	rawCubemapData.resize(6 * width*width*numChannels);
// }

// Cubemap::Cubemap(int width, int height, TextureFormat format)
// {
// 	this->dimension = TextureDimension::Cube;

// 	this->width = width;
// 	this->height = height;
// 	this->format = format;

// 	this->numChannels = calcNumChannels(format);

// 	rawCubemapData.resize(6 * width*width*numChannels);
// }

// Cubemap::~Cubemap()
// {
	
// }

// void Cubemap::generate()
// {
// 	glGenTextures(1, &handle);
// 	glBindTexture(GL_TEXTURE_CUBE_MAP, handle);

// 	GLenum openglFormat = OpenGL::getTextureFormat(format);

// 	for (unsigned int i = 0; i < 6; i++){
// 		glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, openglFormat, width, width, 0, openglFormat, GL_UNSIGNED_BYTE, &rawCubemapData[0]);
// 	}

// 	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
// 	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
// 	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
// 	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
// 	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

// 	glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
// }

// void Cubemap::destroy()
// {
// 	glDeleteTextures(1, &handle);
// }

// void Cubemap::bind()
// {
// 	glBindTexture(GL_TEXTURE_CUBE_MAP, handle);
// }

// void Cubemap::unbind()
// {
// 	glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
// }

// void Cubemap::active(unsigned int slot){
// 	glActiveTexture(GL_TEXTURE0 + slot);
// }

// std::vector<unsigned char> Cubemap::getRawCubemapData()
// {
// 	return rawCubemapData;
// }

// std::vector<Color> Cubemap::getPixels(CubemapFace face)
// {
// 	/*std::vector<Color> colors(width*width*numChannels);*/
// 	std::vector<Color> colors;

// 	int start = (int)face*width*width*numChannels;
// 	int end = start + width*width*numChannels;
// 	for (int i = start; i < end; i += numChannels){
// 		Color color;
// 		if (numChannels == 1){
// 			color.r = rawCubemapData[i];
// 			color.g = rawCubemapData[i];
// 			color.b = rawCubemapData[i];
// 			color.a = 0;
// 		}
// 		else if (numChannels == 3){
// 			color.r = rawCubemapData[i];
// 			color.g = rawCubemapData[i + 1];
// 			color.b = rawCubemapData[i + 2];
// 			color.a = 0;
// 		}
// 		else if (numChannels == 4){
// 			color.r = rawCubemapData[i];
// 			color.g = rawCubemapData[i + 1];
// 			color.b = rawCubemapData[i + 2];
// 			color.a = rawCubemapData[i + 3];
// 		}

// 		colors.push_back(color);
// 	}

// 	return colors;
// }

// Color Cubemap::getPixel(CubemapFace face, int x, int y)
// {
// 	int index =  (int)face*width*width*numChannels + numChannels * (x + width * y);

// 	if (index + numChannels >= rawCubemapData.size()){
// 		Log::Error("Cubemap: pixel index out of range");
// 	}

// 	Color color;
// 	if (numChannels == 1){
// 		color.r = rawCubemapData[index];
// 		color.g = rawCubemapData[index];
// 		color.b = rawCubemapData[index];
// 		color.a = 0;
// 	}
// 	else if (numChannels == 3){
// 		color.r = rawCubemapData[index];
// 		color.g = rawCubemapData[index + 1];
// 		color.b = rawCubemapData[index + 2];
// 		color.a = 0;
// 	}
// 	else if (numChannels == 4){
// 		color.r = rawCubemapData[index];
// 		color.g = rawCubemapData[index + 1];
// 		color.b = rawCubemapData[index + 2];
// 		color.a = rawCubemapData[index + 3];
// 	}

// 	return color;

// }

// void Cubemap::setRawCubemapData(std::vector<unsigned char> data)
// {
// 	if (6*width*height*numChannels != data.size()){
// 		Log::Error("Cubemap: Raw texture data does not match size of cubemap");
// 		return;
// 	}

// 	rawCubemapData = data;
// }

// void Cubemap::setPixels(CubemapFace face, int x, int y, Color color)
// {

// }

// void Cubemap::setPixel(CubemapFace face, int x, int y, Color color)
// {
// 	int index = (int)face*width*width*numChannels + numChannels * (x + width * y);

// 	if (index + numChannels >= rawCubemapData.size()){
// 		Log::Error("Cubemap: pixel index out of range");
// 		return;
// 	}

// 	if (numChannels == 1){
// 		rawCubemapData[index] = color.r;
// 	}
// 	else if (numChannels == 3){
// 		rawCubemapData[index] = color.r;
// 		rawCubemapData[index + 1] = color.g;
// 		rawCubemapData[index + 2] = color.b;
// 	}
// 	else if (numChannels == 4){
// 		rawCubemapData[index] = color.r;
// 		rawCubemapData[index + 1] = color.g;
// 		rawCubemapData[index + 2] = color.b;
// 		rawCubemapData[index + 3] = color.a;
// 	}
// }

// void Cubemap::readPixels()
// {
// 	glBindTexture(GL_TEXTURE_CUBE_MAP, handle);

// 	GLenum openglFormat = OpenGL::getTextureFormat(format);

// 	for (unsigned int i = 0; i < 6; i++){
// 		glGetTexImage(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, openglFormat, GL_UNSIGNED_BYTE, &rawCubemapData[i*width*height*numChannels]);
// 	}

// 	glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
// }

// void Cubemap::apply()
// {
// 	glBindTexture(GL_TEXTURE_CUBE_MAP, handle);

// 	GLenum openglFormat = OpenGL::getTextureFormat(format);
	
// 	for (unsigned int i = 0; i < 6; i++){
// 		glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, openglFormat, width, height, 0, openglFormat, GL_UNSIGNED_BYTE, &rawCubemapData[0]);
// 	}

// 	glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
// }

// GLuint Cubemap::getHandle() const
// {
// 	return handle;
// }