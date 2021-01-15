#include "../../include/core/Texture3D.h"
#include "../../include/core/Log.h"
#include "../../include/graphics/Graphics.h"

using namespace PhysicsEngine;

Texture3D::Texture3D() : Texture()
{
    mDimension = TextureDimension::Tex2D;

    mWidth = 0;
    mHeight = 0;
    mDepth = 0;
    mFormat = TextureFormat::RGB;
    mWrapMode = TextureWrapMode::Repeat;
    mFilterMode = TextureFilterMode::Trilinear;

    mNumChannels = calcNumChannels(mFormat);
    mAnisoLevel = 1;
    mCreated = false;
    mUpdateRequired = false;
}

Texture3D::Texture3D(Guid id) : Texture(id)
{
    mDimension = TextureDimension::Tex2D;

    mWidth = 0;
    mHeight = 0;
    mDepth = 0;
    mFormat = TextureFormat::RGB;
    mWrapMode = TextureWrapMode::Repeat;
    mFilterMode = TextureFilterMode::Trilinear;

    mNumChannels = calcNumChannels(mFormat);
    mAnisoLevel = 1;
    mCreated = false;
    mUpdateRequired = false;
}

Texture3D::Texture3D(int width, int height, int depth, int numChannels) : Texture()
{
    mDimension = TextureDimension::Tex2D;

    mWidth = width;
    mHeight = height;
    mDepth = depth;
    mFormat = TextureFormat::RGB;
    mWrapMode = TextureWrapMode::Repeat;
    mFilterMode = TextureFilterMode::Trilinear;

    mNumChannels = calcNumChannels(mFormat);
    mAnisoLevel = 1;
    mCreated = false;
    mUpdateRequired = false;

    mRawTextureData.resize(width * height * depth * numChannels);
}

Texture3D::~Texture3D()
{
}

std::vector<char> Texture3D::serialize() const
{
    return serialize(mId);
}

std::vector<char> Texture3D::serialize(Guid assetId) const
{
    Texture3DHeader header;
    header.mTextureId = assetId;
    header.mWidth = static_cast<int32_t>(mWidth);
    header.mHeight = static_cast<int32_t>(mHeight);
    header.mDepth = static_cast<int32_t>(mDepth);
    header.mNumChannels = static_cast<int32_t>(mNumChannels);
    header.mAnisoLevel = static_cast<int32_t>(mAnisoLevel);
    header.mDimension = static_cast<uint8_t>(mDimension);
    header.mFormat = static_cast<uint8_t>(mFormat);
    header.mWrapMode = static_cast<uint8_t>(mWrapMode);
    header.mFilterMode = static_cast<uint8_t>(mFilterMode);
    header.mTextureSize = mRawTextureData.size();

    std::size_t len = std::min(size_t(64 - 1), mAssetName.size());
    memcpy(&header.mTextureName[0], &mAssetName[0], len);
    header.mTextureName[len] = '\0';

    size_t numberOfBytes = sizeof(Texture3DHeader) + sizeof(unsigned char) * mRawTextureData.size();

    std::vector<char> data(numberOfBytes);

    size_t start1 = 0;
    size_t start2 = start1 + sizeof(Texture3DHeader);

    memcpy(&data[start1], &header, sizeof(Texture3DHeader));
    memcpy(&data[start2], &mRawTextureData[0], sizeof(unsigned char) * mRawTextureData.size());

    return data;
}

void Texture3D::deserialize(const std::vector<char> &data)
{
    size_t start1 = 0;
    size_t start2 = start1 + sizeof(Texture3DHeader);

    const Texture3DHeader *header = reinterpret_cast<const Texture3DHeader *>(&data[start1]);

    mId = header->mTextureId;
    mAssetName = std::string(header->mTextureName);
    mWidth = static_cast<int>(header->mWidth);
    mHeight = static_cast<int>(header->mHeight);
    mDepth = static_cast<int>(header->mDepth);
    mNumChannels = static_cast<int>(header->mNumChannels);
    mAnisoLevel = static_cast<int>(header->mAnisoLevel);
    mDimension = static_cast<TextureDimension>(header->mDimension);
    mFormat = static_cast<TextureFormat>(header->mFormat);
    mWrapMode = static_cast<TextureWrapMode>(header->mWrapMode);
    mFilterMode = static_cast<TextureFilterMode>(header->mFilterMode);

    mRawTextureData.resize(header->mTextureSize);
    for (size_t i = 0; i < header->mTextureSize; i++)
    {
        mRawTextureData[i] = *reinterpret_cast<const unsigned char *>(&data[start2 + sizeof(unsigned char) * i]);
    }
    this->mCreated = false;
    this->mUpdateRequired = false;
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

void Texture3D::setRawTextureData(const std::vector<unsigned char> &data, int width, int height, int depth,
                                  TextureFormat format)
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

void Texture3D::setPixel(int x, int y, int z, const Color &color)
{
}

void Texture3D::create()
{
    if (mCreated)
    {
        return;
    }

    Graphics::createTexture3D(mFormat, mWrapMode, mFilterMode, mWidth, mHeight, mDepth, mRawTextureData, &mTex);

    mCreated = true;
}

void Texture3D::destroy()
{
    if (!mCreated)
    {
        return;
    }

    Graphics::destroyTexture3D(&mTex);

    mCreated = false;
}

void Texture3D::update()
{
    if (!mUpdateRequired)
    {
        return;
    }

    Graphics::updateTexture3D(mWrapMode, mFilterMode, mAnisoLevel, mTex);

    mUpdateRequired = false;
}

void Texture3D::readPixels()
{
    Graphics::readPixelsTexture3D(mFormat, mWidth, mHeight, mDepth, mNumChannels, mRawTextureData, mTex);
}

void Texture3D::writePixels()
{
    Graphics::writePixelsTexture3D(mFormat, mWidth, mHeight, mDepth, mRawTextureData, mTex);
}
