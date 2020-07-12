#include "../../include/core/Texture3D.h"
#include "../../include/core/Log.h"
#include "../../include/graphics/Graphics.h"

using namespace PhysicsEngine;

Texture3D::Texture3D()
{
	mDimension = TextureDimension::Tex2D;

	mWidth = 0;
	mHeight = 0;
	mDepth = 0;
	mFormat = TextureFormat::RGB;

	mNumChannels = calcNumChannels(mFormat);
	mCreated = false;
}

Texture3D::Texture3D(const std::vector<char>& data)
{
	deserialize(data);
}

Texture3D::Texture3D(int width, int height, int depth, int numChannels)
{
	mDimension = TextureDimension::Tex2D;

	mWidth = width;
	mHeight = height;
	mDepth = depth;
	mFormat = TextureFormat::RGB;

	mNumChannels = calcNumChannels(mFormat);
	mCreated = false;

	mRawTextureData.resize(width * height * depth * numChannels);
}

Texture3D::~Texture3D()
{

}

std::vector<char> Texture3D::serialize() const
{
	return serialize(mAssetId);
}

std::vector<char> Texture3D::serialize(Guid assetId) const
{
	Texture3DHeader header;
	header.mTextureId = assetId;
	header.mWidth = static_cast<int32_t>(mWidth);
	header.mHeight = static_cast<int32_t>(mHeight);
	header.mDepth = static_cast<int32_t>(mDepth);
	header.mNumChannels = static_cast<int32_t>(mNumChannels);
	header.mDimension = static_cast<uint8_t>(mDimension);
	header.mFormat = static_cast<uint8_t>(mFormat);
	header.mTextureSize = mRawTextureData.size();

	size_t numberOfBytes = sizeof(Texture3DHeader) +
		sizeof(unsigned char) * mRawTextureData.size();

	std::vector<char> data(numberOfBytes);

	size_t start1 = 0;
	size_t start2 = start1 + sizeof(Texture3DHeader);

	memcpy(&data[start1], &header, sizeof(Texture3DHeader));
	memcpy(&data[start2], &mRawTextureData[0], sizeof(unsigned char) * mRawTextureData.size());

	return data;
}

void Texture3D::deserialize(const std::vector<char>& data)
{
	size_t start1 = 0;
	size_t start2 = start1 + sizeof(Texture3DHeader);

	const Texture3DHeader* header = reinterpret_cast<const Texture3DHeader*>(&data[start1]);

	mAssetId = header->mTextureId;
	mWidth = static_cast<int>(header->mWidth);
	mHeight = static_cast<int>(header->mHeight);
	mDepth = static_cast<int>(header->mDepth);
	mNumChannels = static_cast<int>(header->mNumChannels);
	mDimension = static_cast<TextureDimension>(header->mDimension);
	mFormat = static_cast<TextureFormat>(header->mFormat);

	mRawTextureData.resize(header->mTextureSize);
	for(size_t i = 0; i < header->mTextureSize; i++){
		mRawTextureData[i] = *reinterpret_cast<const unsigned char*>(&data[start2 + sizeof(unsigned char) * i]);
	}
	this->mCreated = false;
}

int Texture3D::getWidth() const
{
	return mWidth;
}

int Texture3D::getHeight() const
{
	return mHeight;
}

int Texture3D::getDepth() const
{
	return mDepth;
}

void Texture3D::redefine(int width, int height, int depth, TextureFormat format)
{
	mWidth = width;
	mHeight = height;
	mDepth = depth;
	mFormat = TextureFormat::RGB;

	mNumChannels = calcNumChannels(format);
}

std::vector<unsigned char> Texture3D::getRawTextureData() const
{
	return mRawTextureData;
}

Color Texture3D::getPixel(int x, int y, int z) const
{
	return Color::white;
}

void Texture3D::setRawTextureData(std::vector<unsigned char> data, int width, int height, int depth, TextureFormat format)
{
	switch (format)
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
	mDepth = depth;
	mFormat = format;

	mRawTextureData = data;
}

void Texture3D::setPixel(int x, int y, int z, Color color)
{

}

void Texture3D::create()
{
	if (mCreated) {
		return;
	}
	Graphics::create(this, &mTex, &mCreated);
}

void Texture3D::destroy()
{
	if (!mCreated) {
		return;
	}

	Graphics::destroy(this, &mTex, &mCreated);
}

void Texture3D::readPixels()
{
	Graphics::readPixels(this);
}

void Texture3D::apply()
{
	Graphics::apply(this);
}
