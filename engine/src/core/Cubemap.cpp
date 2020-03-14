#include "../../include/core/Cubemap.h"
#include "../../include/core/Log.h"
#include "../../include/graphics/Graphics.h"

using namespace PhysicsEngine;

Cubemap::Cubemap()
{
	this->dimension = TextureDimension::Cube;

	this->width = 0;
	this->format = TextureFormat::RGB;

	this->numChannels = calcNumChannels(format);
	this->created = false;
}

Cubemap::Cubemap(std::vector<char> data)
{
	deserialize(data);
}

Cubemap::Cubemap(int width)
{
	this->dimension = TextureDimension::Cube;

	this->width = width;
	this->format = TextureFormat::RGB;

	this->numChannels = calcNumChannels(format);
	this->created = false;

	rawTextureData.resize(6 * width*width*numChannels);
}

Cubemap::Cubemap(int width, TextureFormat format)
{
	this->dimension = TextureDimension::Cube;

	this->width = width;
	this->format = format;

	this->numChannels = calcNumChannels(format);
	this->created = false;

	rawTextureData.resize(6 * width*width*numChannels);
}

Cubemap::Cubemap(int width, int height, TextureFormat format)
{
	this->dimension = TextureDimension::Cube;

	this->width = width;
	this->format = format;

	this->numChannels = calcNumChannels(format);
	this->created = false;

	rawTextureData.resize(6 * width*width*numChannels);
}

Cubemap::~Cubemap()
{
	
}

std::vector<char> Cubemap::serialize() const
{
	return serialize(assetId);
}

std::vector<char> Cubemap::serialize(Guid assetId) const
{
	CubemapHeader header;
	header.textureId = assetId;
	header.width = width;
	header.numChannels = numChannels;
	header.dimension = dimension;
	header.format = format;
	header.textureSize = rawTextureData.size();

	size_t numberOfBytes = sizeof(CubemapHeader) +
		sizeof(unsigned char) * rawTextureData.size();

	std::vector<char> data(numberOfBytes);

	size_t start1 = 0;
	size_t start2 = start1 + sizeof(CubemapHeader);

	memcpy(&data[start1], &header, sizeof(CubemapHeader));
	memcpy(&data[start2], &rawTextureData[0], sizeof(unsigned char) * rawTextureData.size());

	return data;
}

void Cubemap::deserialize(std::vector<char> data)
{
	size_t start1 = 0;
	size_t start2 = start1 + sizeof(CubemapHeader);

	CubemapHeader* header = reinterpret_cast<CubemapHeader*>(&data[start1]);

	assetId = header->textureId;
	width = header->width;
	numChannels = header->numChannels;
	dimension = static_cast<TextureDimension>(header->dimension);
	format = static_cast<TextureFormat>(header->format);

	rawTextureData.resize(header->textureSize);
	for(size_t i = 0; i < header->textureSize; i++){
		rawTextureData[i] = *reinterpret_cast<unsigned char*>(&data[start2 + sizeof(unsigned char) * i]);
	}
}

int Cubemap::getWidth() const
{
	return width;
}

std::vector<unsigned char> Cubemap::getRawCubemapData() const
{
	return rawTextureData;
}

std::vector<Color> Cubemap::getPixels(CubemapFace face) const
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

Color Cubemap::getPixel(CubemapFace face, int x, int y) const
{
	int index =  (int)face*width*width*numChannels + numChannels * (x + width * y);

	if (index + numChannels >= rawTextureData.size()){
		Log::error("Cubemap: pixel index out of range\n");
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
		Log::error("Cubemap: Raw texture data does not match size of cubemap\n");
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
		Log::error("Cubemap: pixel index out of range\n");
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

void Cubemap::create()
{
	if (created) {
		return;
	}
	Graphics::create(this, &tex, &created);
}

void Cubemap::destroy()
{
	if (!created) {
		return;
	}

	Graphics::destroy(this, &tex, &created);
}

void Cubemap::readPixels()
{
	Graphics::readPixels(this);
}

void Cubemap::apply()
{
	Graphics::apply(this);
}