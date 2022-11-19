#include "../../include/core/Cubemap.h"
#include "../../include/core/Log.h"
#include "../../include/graphics/Renderer.h"

using namespace PhysicsEngine;

Cubemap::Cubemap(World *world, const Id &id) : Texture(world, id)
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

Cubemap::Cubemap(World *world, const Guid &guid, const Id &id) : Texture(world, guid, id)
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

Cubemap::Cubemap(World *world, const Id &id, int width) : Texture(world, id)
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

Cubemap::Cubemap(World *world, const Id &id, int width, TextureFormat format) : Texture(world, id)
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

void Cubemap::serialize(YAML::Node &out) const
{
    Texture::serialize(out);

    out["width"] = mWidth;
}

void Cubemap::deserialize(const YAML::Node &in)
{
    Texture::deserialize(in);

    mWidth = YAML::getValue<int>(in, "width");
}

int Cubemap::getType() const
{
    return PhysicsEngine::CUBEMAP_TYPE;
}

std::string Cubemap::getObjectName() const
{
    return PhysicsEngine::CUBEMAP_NAME;
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
            color.mR = mRawTextureData[i];
            color.mG = mRawTextureData[i];
            color.mB = mRawTextureData[i];
            color.mA = 0;
        }
        else if (mNumChannels == 3)
        {
            color.mR = mRawTextureData[i];
            color.mG = mRawTextureData[i + 1];
            color.mB = mRawTextureData[i + 2];
            color.mA = 0;
        }
        else if (mNumChannels == 4)
        {
            color.mR = mRawTextureData[i];
            color.mG = mRawTextureData[i + 1];
            color.mB = mRawTextureData[i + 2];
            color.mA = mRawTextureData[i + 3];
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
        color.mR = mRawTextureData[index];
        color.mG = mRawTextureData[index];
        color.mB = mRawTextureData[index];
        color.mA = 0;
    }
    else if (mNumChannels == 3)
    {
        color.mR = mRawTextureData[index];
        color.mG = mRawTextureData[index + 1];
        color.mB = mRawTextureData[index + 2];
        color.mA = 0;
    }
    else if (mNumChannels == 4)
    {
        color.mR = mRawTextureData[index];
        color.mG = mRawTextureData[index + 1];
        color.mB = mRawTextureData[index + 2];
        color.mA = mRawTextureData[index + 3];
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
        mRawTextureData[index] = color.mR;
    }
    else if (mNumChannels == 3)
    {
        mRawTextureData[index] = color.mR;
        mRawTextureData[index + 1] = color.mG;
        mRawTextureData[index + 2] = color.mB;
    }
    else if (mNumChannels == 4)
    {
        mRawTextureData[index] = color.mR;
        mRawTextureData[index + 1] = color.mG;
        mRawTextureData[index + 2] = color.mB;
        mRawTextureData[index + 3] = color.mA;
    }
}

void Cubemap::create()
{
    if (mCreated)
    {
        return;
    }

    Renderer::getRenderer()->createCubemap(mFormat, mWrapMode, mFilterMode, mWidth, mRawTextureData, &mTex);

    mCreated = true;
}

void Cubemap::destroy()
{
    if (!mCreated)
    {
        return;
    }

    Renderer::getRenderer()->destroyCubemap(&mTex);

    mCreated = false;
}

void Cubemap::update()
{
    if (!mUpdateRequired)
    {
        return;
    }

    Renderer::getRenderer()->updateCubemap(mWrapMode, mFilterMode, mAnisoLevel, mTex);

    mUpdateRequired = false;
}

void Cubemap::readPixels()
{
    Renderer::getRenderer()->readPixelsCubemap(mFormat, mWidth, mNumChannels, mRawTextureData, mTex);
}

void Cubemap::writePixels()
{
    Renderer::getRenderer()->writePixelsCubemap(mFormat, mWidth, mRawTextureData, mTex);
}