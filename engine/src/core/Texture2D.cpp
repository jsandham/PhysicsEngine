#include "../../include/core/Texture2D.h"
#include "../../include/core/Log.h"
#include "../../include/graphics/Graphics.h"
#include "../../include/stb_image/stb_image.h"

using namespace PhysicsEngine;

Texture2D::Texture2D()
{
	mDimension = TextureDimension::Tex2D;

	mWidth = 0;
	mHeight = 0;
	mFormat = TextureFormat::RGB;

	mNumChannels = calcNumChannels(mFormat);
	mCreated = false;
}

Texture2D::Texture2D(std::vector<char> data)
{
	deserialize(data);
}

Texture2D::Texture2D(int width, int height)
{
	mDimension = TextureDimension::Tex2D;

	mWidth = width;
	mHeight = height;
	mFormat = TextureFormat::RGB;

	mNumChannels = calcNumChannels(mFormat);
	mCreated = false;

	mRawTextureData.resize(width*height*mNumChannels);
}

Texture2D::Texture2D(int width, int height, TextureFormat format)
{
	mDimension = TextureDimension::Tex2D;

	mWidth = width;
	mHeight = height;
	mFormat = format;

	mNumChannels = calcNumChannels(format);
	mCreated = false;

	mRawTextureData.resize(width*height*mNumChannels);
}

Texture2D::~Texture2D()
{
	
}

std::vector<char> Texture2D::serialize() const
{
	return serialize(mAssetId);
}

std::vector<char> Texture2D::serialize(Guid assetId) const
{
	Texture2DHeader header;
	header.mTextureId = assetId;
	header.mWidth = mWidth;
	header.mHeight = mHeight;
	header.mNumChannels = mNumChannels;
	header.mDimension = mDimension;
	header.mFormat = mFormat;
	header.mTextureSize = mRawTextureData.size();

	size_t numberOfBytes = sizeof(Texture2DHeader) +
		sizeof(unsigned char) * mRawTextureData.size();

	std::vector<char> data(numberOfBytes);

	size_t start1 = 0;
	size_t start2 = start1 + sizeof(Texture2DHeader);

	memcpy(&data[start1], &header, sizeof(Texture2DHeader));
	memcpy(&data[start2], &mRawTextureData[0], sizeof(unsigned char) * mRawTextureData.size());

	return data;
}

void Texture2D::deserialize(std::vector<char> data)
{
	size_t start1 = 0;
	size_t start2 = start1 + sizeof(Texture2DHeader);

	Texture2DHeader* header = reinterpret_cast<Texture2DHeader*>(&data[start1]);

	mAssetId = header->mTextureId;
	mWidth = header->mWidth;
	mHeight = header->mHeight;
	mNumChannels = header->mNumChannels;
	mDimension = static_cast<TextureDimension>(header->mDimension);
	mFormat = static_cast<TextureFormat>(header->mFormat);

	mRawTextureData.resize(header->mTextureSize);
	for(size_t i = 0; i < header->mTextureSize; i++){
		mRawTextureData[i] = *reinterpret_cast<unsigned char*>(&data[start2 + sizeof(unsigned char) * i]);
	}

	mCreated = false;
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
	return mWidth;
}

int Texture2D::getHeight() const
{
	return mHeight;
}

void Texture2D::redefine(int width, int height, TextureFormat format)
{
	mWidth = width;
	mHeight = height;
	mFormat = format;

	mNumChannels = calcNumChannels(format);

	mRawTextureData.resize(width*height*mNumChannels);
}

std::vector<unsigned char> Texture2D::getRawTextureData() const
{
	return mRawTextureData;
}

std::vector<Color32> Texture2D::getPixels() const
{
	std::vector<Color32> colors;

	colors.resize(mWidth*mHeight);

	for (unsigned int i = 0; i < colors.size(); i++){
		colors[i].r = mRawTextureData[mNumChannels * i];
		colors[i].g = mRawTextureData[mNumChannels * i + 1];
		colors[i].b = mRawTextureData[mNumChannels * i + 2];
	}

	return colors;
}

Color32 Texture2D::getPixel(int x, int y) const
{
	int index = mNumChannels * (x + mWidth * y);

	Color32 color;

	if (index + mNumChannels >= mRawTextureData.size()){
		Log::error("Texture2D: pixel index out of range\n");
		return color;
	}

	if (mNumChannels == 1){
		color.r = mRawTextureData[index];
		color.g = mRawTextureData[index];
		color.b = mRawTextureData[index];
		color.a = 0;
	}
	else if (mNumChannels == 3){
		color.r = mRawTextureData[index];
		color.g = mRawTextureData[index + 1];
		color.b = mRawTextureData[index + 2];
		color.a = 0;
	}
	else if (mNumChannels == 4){
		color.r = mRawTextureData[index];
		color.g = mRawTextureData[index + 1];
		color.b = mRawTextureData[index + 2];
		color.a = mRawTextureData[index + 3];
	}

	return color;
}

void Texture2D::setRawTextureData(std::vector<unsigned char> data, int width, int height, TextureFormat format)
{
	switch(format)
	{
		case TextureFormat::Depth:
			mNumChannels = 1;
			break;
		case TextureFormat::RG:
			mNumChannels = 2;
			break;
		case TextureFormat::RGB:
			mNumChannels = 3;
			break;
		case TextureFormat::RGBA:
			mNumChannels = 4;
			break;
		default:
			Log::error("Unsupported texture format %d\n", format);
			return;
	}

	mWidth = width;
	mHeight = height;
	mFormat = format;

	mRawTextureData = data;
}

void Texture2D::setPixels(std::vector<Color32> colors)
{
	if (colors.size() != mWidth*mHeight){
		Log::error("Texture2D: error when trying to set pixels. Number of colors must match dimensions of texture\n");
		return;
	}

	for (size_t i = 0; i < colors.size(); i++){
		if (mNumChannels == 1){
			mRawTextureData[i] = colors[i].r;
		}
		else if (mNumChannels == 3){
			mRawTextureData[i] = colors[i].r;
			mRawTextureData[i + 1] = colors[i].g;
			mRawTextureData[i + 2] = colors[i].b;
		}
		else if (mNumChannels == 4){
			mRawTextureData[i] = colors[i].r;
			mRawTextureData[i + 1] = colors[i].g;
			mRawTextureData[i + 2] = colors[i].b;
			mRawTextureData[i + 3] = colors[i].a;
		}
	}
}

void Texture2D::setPixel(int x, int y, Color32 color)
{
	int index = mNumChannels * (x + mWidth * y);

	if (index + mNumChannels >= mRawTextureData.size()){
		Log::error("Texture2D: pixel index out of range\n");
		return;
	}

	if (mNumChannels == 1){
		mRawTextureData[index] = color.r;
	}
	else if (mNumChannels == 3){
		mRawTextureData[index] = color.r;
		mRawTextureData[index + 1] = color.g;
		mRawTextureData[index + 2] = color.b;
	}
	else if (mNumChannels == 4){
		mRawTextureData[index] = color.r;
		mRawTextureData[index + 1] = color.g;
		mRawTextureData[index + 2] = color.b;
		mRawTextureData[index + 3] = color.a;
	}
}

void Texture2D::create()
{
	if (mCreated) {
		return;
	}
	Graphics::create(this, &mTex, &mCreated);
}

void Texture2D::destroy()
{
	if (!mCreated) {
		return;
	}

	Graphics::destroy(this, &mTex, &mCreated);
}

void Texture2D::readPixels()
{
	Graphics::readPixels(this);
}

void Texture2D::apply()
{
	Graphics::apply(this);
}