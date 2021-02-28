#include "../../include/core/Cubemap.h"
#include "../../include/core/Log.h"
#include "../../include/core/Serialization.h"
#include "../../include/graphics/Graphics.h"

using namespace PhysicsEngine;

Cubemap::Cubemap() : Texture()
{
    mDimension = TextureDimension::Cube;

    mWidth = 0;
    mFormat = TextureFormat::RGB;
    mWrapMode = TextureWrapMode::Repeat;
    mFilterMode = TextureFilterMode::Trilinear;

    mNumChannels = calcNumChannels(mFormat);
    mAnisoLevel = 1;
    mCreated = false;
    mUpdateRequired = false;
}

Cubemap::Cubemap(Guid id) : Texture(id)
{
    mDimension = TextureDimension::Cube;

    mWidth = 0;
    mFormat = TextureFormat::RGB;
    mWrapMode = TextureWrapMode::Repeat;
    mFilterMode = TextureFilterMode::Trilinear;

    mNumChannels = calcNumChannels(mFormat);
    mAnisoLevel = 1;
    mCreated = false;
    mUpdateRequired = false;
}

Cubemap::Cubemap(int width) : Texture()
{
    mDimension = TextureDimension::Cube;

    mWidth = width;
    mFormat = TextureFormat::RGB;
    mWrapMode = TextureWrapMode::Repeat;
    mFilterMode = TextureFilterMode::Trilinear;

    mNumChannels = calcNumChannels(mFormat);
    mAnisoLevel = 1;
    mCreated = false;
    mUpdateRequired = false;

    mRawTextureData.resize(6 * width * width * mNumChannels);
}

Cubemap::Cubemap(int width, TextureFormat format) : Texture()
{
    mDimension = TextureDimension::Cube;

    mWidth = width;
    mFormat = format;
    mWrapMode = TextureWrapMode::Repeat;
    mFilterMode = TextureFilterMode::Trilinear;

    mNumChannels = calcNumChannels(format);
    mAnisoLevel = 1;
    mCreated = false;
    mUpdateRequired = false;

    mRawTextureData.resize(6 * width * width * mNumChannels);
}

Cubemap::~Cubemap()
{
}

void Cubemap::serialize(std::ostream &out) const
{
    Texture::serialize(out);

    PhysicsEngine::write<int>(out, mWidth);
}

void Cubemap::deserialize(std::istream &in)
{
    Texture::deserialize(in);

    PhysicsEngine::read<int>(in, mWidth);
}

void Cubemap::serialize(YAML::Node& out) const
{
    Texture::serialize(out);

    out["width"] = mWidth;
}

void Cubemap::deserialize(const YAML::Node& in)
{
    Texture::deserialize(in);

    mWidth = in["width"].as<int>();
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

    int start = (int)face * mWidth * mWidth * mNumChannels;
    int end = start + mWidth * mWidth * mNumChannels;
    for (int i = start; i < end; i += mNumChannels)
    {
        Color32 color;
        if (mNumChannels == 1)
        {
            color.r = mRawTextureData[i];
            color.g = mRawTextureData[i];
            color.b = mRawTextureData[i];
            color.a = 0;
        }
        else if (mNumChannels == 3)
        {
            color.r = mRawTextureData[i];
            color.g = mRawTextureData[i + 1];
            color.b = mRawTextureData[i + 2];
            color.a = 0;
        }
        else if (mNumChannels == 4)
        {
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
    // clamp x and y
    x = std::max(0, std::min(mWidth, x));
    y = std::max(0, std::min(mWidth, y));

    int index = static_cast<int>(face) * mWidth * mWidth * mNumChannels + mNumChannels * (x + mWidth * y);

    int size = static_cast<int>(mRawTextureData.size());
    if (index + mNumChannels >= size)
    {
        Log::error("Cubemap: pixel index out of range\n");
    }

    Color32 color;
    if (mNumChannels == 1)
    {
        color.r = mRawTextureData[index];
        color.g = mRawTextureData[index];
        color.b = mRawTextureData[index];
        color.a = 0;
    }
    else if (mNumChannels == 3)
    {
        color.r = mRawTextureData[index];
        color.g = mRawTextureData[index + 1];
        color.b = mRawTextureData[index + 2];
        color.a = 0;
    }
    else if (mNumChannels == 4)
    {
        color.r = mRawTextureData[index];
        color.g = mRawTextureData[index + 1];
        color.b = mRawTextureData[index + 2];
        color.a = mRawTextureData[index + 3];
    }

    return color;
}

void Cubemap::setRawCubemapData(const std::vector<unsigned char> &data)
{
    int size = static_cast<int>(mRawTextureData.size());
    if (6 * mWidth * mWidth * mNumChannels != size)
    {
        Log::error("Cubemap: Raw texture data does not match size of cubemap\n");
        return;
    }

    mRawTextureData = data;
}

void Cubemap::setPixels(CubemapFace face, int x, int y, const Color32 &color)
{
}

void Cubemap::setPixel(CubemapFace face, int x, int y, const Color32 &color)
{
    // clamp x and y
    x = std::max(0, std::min(mWidth, x));
    y = std::max(0, std::min(mWidth, y));

    int index = static_cast<int>(face) * mWidth * mWidth * mNumChannels + mNumChannels * (x + mWidth * y);

    int size = static_cast<int>(mRawTextureData.size());
    if (index + mNumChannels >= size)
    {
        Log::error("Cubemap: pixel index out of range\n");
        return;
    }

    if (mNumChannels == 1)
    {
        mRawTextureData[index] = color.r;
    }
    else if (mNumChannels == 3)
    {
        mRawTextureData[index] = color.r;
        mRawTextureData[index + 1] = color.g;
        mRawTextureData[index + 2] = color.b;
    }
    else if (mNumChannels == 4)
    {
        mRawTextureData[index] = color.r;
        mRawTextureData[index + 1] = color.g;
        mRawTextureData[index + 2] = color.b;
        mRawTextureData[index + 3] = color.a;
    }
}

void Cubemap::create()
{
    if (mCreated)
    {
        return;
    }

    Graphics::createCubemap(mFormat, mWrapMode, mFilterMode, mWidth, mRawTextureData, &mTex);

    mCreated = true;
}

void Cubemap::destroy()
{
    if (!mCreated)
    {
        return;
    }

    Graphics::destroyCubemap(&mTex);

    mCreated = false;
}

void Cubemap::update()
{
    if (!mUpdateRequired)
    {
        return;
    }

    Graphics::updateCubemap(mWrapMode, mFilterMode, mAnisoLevel, mTex);

    mUpdateRequired = false;
}

void Cubemap::readPixels()
{
    Graphics::readPixelsCubemap(mFormat, mWidth, mNumChannels, mRawTextureData, mTex);
}

void Cubemap::writePixels()
{
    Graphics::writePixelsCubemap(mFormat, mWidth, mRawTextureData, mTex);
}