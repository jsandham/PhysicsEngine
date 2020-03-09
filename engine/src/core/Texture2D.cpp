#include <iostream>

#include "../../include/core/Texture2D.h"
#include "../../include/core/Log.h"
#include "../../include/graphics/Graphics.h"
#include "../../include/stb_image/stb_image.h"

using namespace PhysicsEngine;

Texture2D::Texture2D()
{
	this->dimension = TextureDimension::Tex2D;

	this->width = 0;
	this->height = 0;
	this->format = TextureFormat::RGB;

	this->numChannels = calcNumChannels(format);
	this->isCreated = false;
}

Texture2D::Texture2D(std::vector<char> data)
{
	deserialize(data);
}

Texture2D::Texture2D(int width, int height)
{
	this->dimension = TextureDimension::Tex2D;

	this->width = width;
	this->height = height;
	this->format = TextureFormat::RGB;

	this->numChannels = calcNumChannels(format);
	this->isCreated = false;

	rawTextureData.resize(width*height*numChannels);
}

Texture2D::Texture2D(int width, int height, TextureFormat format)
{
	this->dimension = TextureDimension::Tex2D;

	this->width = width;
	this->height = height;
	this->format = format;

	this->numChannels = calcNumChannels(format);
	this->isCreated = false;

	rawTextureData.resize(width*height*numChannels);
}

Texture2D::~Texture2D()
{
	
}

std::vector<char> Texture2D::serialize() const
{
	return serialize(assetId);
}

std::vector<char> Texture2D::serialize(Guid assetId) const
{
	Texture2DHeader header;
	header.textureId = assetId;
	header.width = width;
	header.height = height;
	header.numChannels = numChannels;
	header.dimension = dimension;
	header.format = format;
	header.textureSize = rawTextureData.size();

	size_t numberOfBytes = sizeof(Texture2DHeader) +
		sizeof(unsigned char) * rawTextureData.size();

	std::vector<char> data(numberOfBytes);

	size_t start1 = 0;
	size_t start2 = start1 + sizeof(Texture2DHeader);

	memcpy(&data[start1], &header, sizeof(Texture2DHeader));
	memcpy(&data[start2], &rawTextureData[0], sizeof(unsigned char) * rawTextureData.size());

	return data;
}

void Texture2D::deserialize(std::vector<char> data)
{
	size_t start1 = 0;
	size_t start2 = start1 + sizeof(Texture2DHeader);

	Texture2DHeader* header = reinterpret_cast<Texture2DHeader*>(&data[start1]);

	assetId = header->textureId;
	width = header->width;
	height = header->height;
	numChannels = header->numChannels;
	dimension = static_cast<TextureDimension>(header->dimension);
	format = static_cast<TextureFormat>(header->format);

	rawTextureData.resize(header->textureSize);
	for(size_t i = 0; i < header->textureSize; i++){
		rawTextureData[i] = *reinterpret_cast<unsigned char*>(&data[start2 + sizeof(unsigned char) * i]);
	}

	this->isCreated = false;
}

void Texture2D::load(const std::string& filepath)
{
	stbi_set_flip_vertically_on_load(true);

	int width, height, numChannels;
	unsigned char* raw = stbi_load(filepath.c_str(), &width, &height, &numChannels, 0);

	if (raw != NULL) {
		TextureFormat format;
		switch (numChannels)
		{
		case 1:
			format = TextureFormat::Depth;
			break;
		case 2:
			format = TextureFormat::RG;
			break;
		case 3:
			format = TextureFormat::RGB;
			break;
		case 4:
			format = TextureFormat::RGBA;
			break;
		default:
			std::string message = "Error: Unsupported number of channels (" + std::to_string(numChannels) + ") found when loading texture " + filepath + "\n";
			Log::error(message.c_str());
			return;
		}

		std::vector<unsigned char> data;
		data.resize(width * height * numChannels);

		for (unsigned int j = 0; j < data.size(); j++) { data[j] = raw[j]; }

		stbi_image_free(raw);

		setRawTextureData(data, width, height, format);
	}
	else {
		std::string message = "Error: stbi_load failed to load texture " + filepath + " with reported reason: " + stbi_failure_reason() + "\n";
		Log::error(message.c_str());
		return;
	}
}

int Texture2D::getWidth() const
{
	return width;
}

int Texture2D::getHeight() const
{
	return height;
}

void Texture2D::redefine(int width, int height, TextureFormat format)
{
	this->width = width;
	this->height = height;
	this->format = format;

	this->numChannels = calcNumChannels(format);

	rawTextureData.resize(width*height*numChannels);
}

std::vector<unsigned char> Texture2D::getRawTextureData() const
{
	return rawTextureData;
}

std::vector<Color> Texture2D::getPixels() const
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

Color Texture2D::getPixel(int x, int y) const
{
	int index = numChannels * (x + width * y);

	Color color;

	if (index + numChannels >= rawTextureData.size()){
		Log::error("Texture2D: pixel index out of range\n");
		return color;
	}

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

TextureFormat Texture2D::getFormat() const
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
			Log::error("Unsupported texture format %d\n", format);
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
		Log::error("Texture2D: error when trying to set pixels. Number of colors must match dimensions of texture\n");
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
		Log::error("Texture2D: pixel index out of range\n");
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