#include "../../include/core/Cubemap.h"
#include "../../include/core/Log.h"
#include "../../include/graphics/Graphics.h"

using namespace PhysicsEngine;

Cubemap::Cubemap()
{
	mDimension = TextureDimension::Cube;

	mWidth = 0;
	mFormat = TextureFormat::RGB;

	mNumChannels = calcNumChannels(mFormat);
	mCreated = false;
}

Cubemap::Cubemap(const std::vector<char>& data)
{
	deserialize(data);
}

Cubemap::Cubemap(int width)
{
	mDimension = TextureDimension::Cube;

	mWidth = width;
	mFormat = TextureFormat::RGB;

	mNumChannels = calcNumChannels(mFormat);
	mCreated = false;

	mRawTextureData.resize(6 * width*width*mNumChannels);
}

Cubemap::Cubemap(int width, TextureFormat format)
{
	mDimension = TextureDimension::Cube;

	mWidth = width;
	mFormat = format;

	mNumChannels = calcNumChannels(format);
	mCreated = false;

	mRawTextureData.resize(6 * width*width*mNumChannels);
}

Cubemap::Cubemap(int width, int height, TextureFormat format)
{
	mDimension = TextureDimension::Cube;

	mWidth = width;
	mFormat = format;

	mNumChannels = calcNumChannels(format);
	mCreated = false;

	mRawTextureData.resize(6 * width*width*mNumChannels);
}

Cubemap::~Cubemap()
{
	
}

std::vector<char> Cubemap::serialize() const
{
	return serialize(mAssetId);
}

std::vector<char> Cubemap::serialize(Guid assetId) const
{
	CubemapHeader header;
	header.mTextureId = assetId;
	header.mWidth = mWidth;
	header.mNumChannels = mNumChannels;
	header.mDimension = mDimension;
	header.mFormat = mFormat;
	header.mTextureSize = mRawTextureData.size();

	size_t numberOfBytes = sizeof(CubemapHeader) +
		sizeof(unsigned char) * mRawTextureData.size();

	std::vector<char> data(numberOfBytes);

	size_t start1 = 0;
	size_t start2 = start1 + sizeof(CubemapHeader);

	memcpy(&data[start1], &header, sizeof(CubemapHeader));
	memcpy(&data[start2], &mRawTextureData[0], sizeof(unsigned char) * mRawTextureData.size());

	return data;
}

void Cubemap::deserialize(const std::vector<char>& data)
{
	size_t start1 = 0;
	size_t start2 = start1 + sizeof(CubemapHeader);

	const CubemapHeader* header = reinterpret_cast<const CubemapHeader*>(&data[start1]);

	mAssetId = header->mTextureId;
	mWidth = header->mWidth;
	mNumChannels = header->mNumChannels;
	mDimension = static_cast<TextureDimension>(header->mDimension);
	mFormat = static_cast<TextureFormat>(header->mFormat);

	mRawTextureData.resize(header->mTextureSize);
	for(size_t i = 0; i < header->mTextureSize; i++){
		mRawTextureData[i] = *reinterpret_cast<const unsigned char*>(&data[start2 + sizeof(unsigned char) * i]);
	}
}

int Cubemap::getWidth() const
{
	return mWidth;
}

std::vector<unsigned char> Cubemap::getRawCubemapData() const
{
	return mRawTextureData;
}

std::vector<Color32> Cubemap::getPixels(CubemapFace face) const
{
	/*std::vector<Color> colors(width*width*numChannels);*/
	std::vector<Color32> colors;

	int start = (int)face*mWidth * mWidth * mNumChannels;
	int end = start + mWidth * mWidth * mNumChannels;
	for (int i = start; i < end; i += mNumChannels){
		Color32 color;
		if (mNumChannels == 1){
			color.r = mRawTextureData[i];
			color.g = mRawTextureData[i];
			color.b = mRawTextureData[i];
			color.a = 0;
		}
		else if (mNumChannels == 3){
			color.r = mRawTextureData[i];
			color.g = mRawTextureData[i + 1];
			color.b = mRawTextureData[i + 2];
			color.a = 0;
		}
		else if (mNumChannels == 4){
			color.r = mRawTextureData[i];
			color.g = mRawTextureData[i + 1];
			color.b = mRawTextureData[i + 2];
			color.a = mRawTextureData[i + 3];
		}

		colors.push_back(color);
	}

	return colors;
}

Color32 Cubemap::getPixel(CubemapFace face, int x, int y) const
{
	int index =  (int)face * mWidth * mWidth * mNumChannels + mNumChannels * (x + mWidth * y);

	if (index + mNumChannels >= mRawTextureData.size()){
		Log::error("Cubemap: pixel index out of range\n");
	}

	Color32 color;
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

void Cubemap::setRawCubemapData(std::vector<unsigned char> data)
{
	if (6 * mWidth * mWidth * mNumChannels != data.size()){
		Log::error("Cubemap: Raw texture data does not match size of cubemap\n");
		return;
	}

	mRawTextureData = data;
}

void Cubemap::setPixels(CubemapFace face, int x, int y, Color32 color)
{

}

void Cubemap::setPixel(CubemapFace face, int x, int y, Color32 color)
{
	int index = (int)face * mWidth * mWidth * mNumChannels + mNumChannels * (x + mWidth * y);

	if (index + mNumChannels >= mRawTextureData.size()){
		Log::error("Cubemap: pixel index out of range\n");
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

void Cubemap::create()
{
	if (mCreated) {
		return;
	}
	Graphics::create(this, &mTex, &mCreated);
}

void Cubemap::destroy()
{
	if (!mCreated) {
		return;
	}

	Graphics::destroy(this, &mTex, &mCreated);
}

void Cubemap::readPixels()
{
	Graphics::readPixels(this);
}

void Cubemap::apply()
{
	Graphics::apply(this);
}