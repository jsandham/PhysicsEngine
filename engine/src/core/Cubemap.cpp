#include "../../include/core/Cubemap.h"
#include "../../include/core/Log.h"
#include "../../include/core/Texture2D.h"
#include "../../include/core/World.h"
#include "../../include/graphics/Renderer.h"

using namespace PhysicsEngine;

Cubemap::Cubemap(World *world, const Id &id) : Texture(world, id)
{
    mLeftTexGuid = Guid::INVALID;
    mRightTexGuid = Guid::INVALID;
    mBottomTexGuid = Guid::INVALID;
    mTopTexGuid = Guid::INVALID;
    mBackTexGuid = Guid::INVALID;
    mFrontTexGuid = Guid::INVALID;

    mDimension = TextureDimension::Cube;

    mWidth = 0;
    mFormat = TextureFormat::RGB;
    mWrapMode = TextureWrapMode::Repeat;
    mFilterMode = TextureFilterMode::Trilinear;

    mNumChannels = calcNumChannels(mFormat);
    mAnisoLevel = 1;
    mDeviceUpdateRequired = false;
    mUpdateRequired = false;

    mCube = CubemapHandle::create(mWidth, mFormat, mWrapMode, mFilterMode);
}

Cubemap::Cubemap(World *world, const Guid &guid, const Id &id) : Texture(world, guid, id)
{
    mLeftTexGuid = Guid::INVALID;
    mRightTexGuid = Guid::INVALID;
    mBottomTexGuid = Guid::INVALID;
    mTopTexGuid = Guid::INVALID;
    mBackTexGuid = Guid::INVALID;
    mFrontTexGuid = Guid::INVALID;

    mDimension = TextureDimension::Cube;

    mWidth = 0;
    mFormat = TextureFormat::RGB;
    mWrapMode = TextureWrapMode::Repeat;
    mFilterMode = TextureFilterMode::Trilinear;

    mNumChannels = calcNumChannels(mFormat);
    mAnisoLevel = 1;
    mDeviceUpdateRequired = false;
    mUpdateRequired = false;

    mCube = CubemapHandle::create(mWidth, mFormat, mWrapMode, mFilterMode);
}

Cubemap::Cubemap(World *world, const Id &id, int width) : Texture(world, id)
{
    mLeftTexGuid = Guid::INVALID;
    mRightTexGuid = Guid::INVALID;
    mBottomTexGuid = Guid::INVALID;
    mTopTexGuid = Guid::INVALID;
    mBackTexGuid = Guid::INVALID;
    mFrontTexGuid = Guid::INVALID;

    mDimension = TextureDimension::Cube;

    mWidth = width;
    mFormat = TextureFormat::RGB;
    mWrapMode = TextureWrapMode::Repeat;
    mFilterMode = TextureFilterMode::Trilinear;

    mNumChannels = calcNumChannels(mFormat);
    mAnisoLevel = 1;
    mDeviceUpdateRequired = false;
    mUpdateRequired = false;

    mRawTextureData.resize(6 * width * width * mNumChannels);

    mCube = CubemapHandle::create(mWidth, mFormat, mWrapMode, mFilterMode);
}

Cubemap::Cubemap(World *world, const Id &id, int width, TextureFormat format) : Texture(world, id)
{
    mLeftTexGuid = Guid::INVALID;
    mRightTexGuid = Guid::INVALID;
    mBottomTexGuid = Guid::INVALID;
    mTopTexGuid = Guid::INVALID;
    mBackTexGuid = Guid::INVALID;
    mFrontTexGuid = Guid::INVALID;

    mDimension = TextureDimension::Cube;

    mWidth = width;
    mFormat = format;
    mWrapMode = TextureWrapMode::Repeat;
    mFilterMode = TextureFilterMode::Trilinear;

    mNumChannels = calcNumChannels(format);
    mAnisoLevel = 1;
    mDeviceUpdateRequired = false;
    mUpdateRequired = false;

    mRawTextureData.resize(6 * width * width * mNumChannels);

    mCube = CubemapHandle::create(mWidth, mFormat, mWrapMode, mFilterMode);
}

Cubemap::~Cubemap()
{
    delete mCube;
}

void Cubemap::serialize(YAML::Node &out) const
{
    Texture::serialize(out);

    out["leftTexId"] = mLeftTexGuid;
    out["rightTexId"] = mRightTexGuid;
    out["bottomTexId"] = mBottomTexGuid;
    out["topTexId"] = mTopTexGuid;
    out["backTexId"] = mBackTexGuid;
    out["frontTexId"] = mFrontTexGuid;

    out["width"] = mWidth;
}

void Cubemap::deserialize(const YAML::Node &in)
{
    Texture::deserialize(in);

    mLeftTexGuid = YAML::getValue<Guid>(in, "leftTexId");
    mRightTexGuid = YAML::getValue<Guid>(in, "rightTexId");
    mBottomTexGuid = YAML::getValue<Guid>(in, "bottomTexId");
    mTopTexGuid = YAML::getValue<Guid>(in, "topTexId");
    mBackTexGuid = YAML::getValue<Guid>(in, "backTexId");
    mFrontTexGuid = YAML::getValue<Guid>(in, "frontTexId");

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

Guid Cubemap::getTexId(CubemapFace face) const
{
    switch (face)
    {
    case CubemapFace::NegativeX:
        return mLeftTexGuid;
    case CubemapFace::PositiveX:
        return mRightTexGuid;
    case CubemapFace::NegativeY:
        return mBottomTexGuid;
    case CubemapFace::PositiveY:
        return mTopTexGuid;
    case CubemapFace::NegativeZ:
        return mBackTexGuid;
    case CubemapFace::PositiveZ:
        return mFrontTexGuid;
    }

    return Guid::INVALID;
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
    mDeviceUpdateRequired = true;
}

void Cubemap::setRawCubemapData(CubemapFace face, const std::vector<unsigned char> &data)
{
    int size = static_cast<int>(mRawTextureData.size());
    if (mWidth * mWidth * mNumChannels != size)
    {
        Log::error("Cubemap: Raw texture data does not match size of cubemap\n");
        return;
    }

    size_t offset = static_cast<int>(face) * mWidth * mWidth * mNumChannels;

    for (size_t i = 0; i < data.size(); i++)
    {
        mRawTextureData[offset + i] = data[i];
    }

    mDeviceUpdateRequired = true;
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

    mDeviceUpdateRequired = true;
}

void Cubemap::setTexId(CubemapFace face, const Guid &texId)
{
    switch (face)
    {
    case CubemapFace::NegativeX:
        mLeftTexGuid = texId;
        break;
    case CubemapFace::PositiveX:
        mRightTexGuid = texId;
        break;
    case CubemapFace::NegativeY:
        mBottomTexGuid = texId;
        break;
    case CubemapFace::PositiveY:
        mTopTexGuid = texId;
        break;
    case CubemapFace::NegativeZ:
        mBackTexGuid = texId;
        break;
    case CubemapFace::PositiveZ:
        mFrontTexGuid = texId;
        break;
    }
}

void Cubemap::fillCubemapFromAttachedTexture(CubemapFace face)
{
    Texture2D *texture = nullptr;
    switch (face)
    {
    case CubemapFace::NegativeX:
        texture = mWorld->getAssetByGuid<Texture2D>(mLeftTexGuid);
        break;
    case CubemapFace::PositiveX:
        texture = mWorld->getAssetByGuid<Texture2D>(mRightTexGuid);
        break;
    case CubemapFace::NegativeY:
        texture = mWorld->getAssetByGuid<Texture2D>(mBottomTexGuid);
        break;
    case CubemapFace::PositiveY:
        texture = mWorld->getAssetByGuid<Texture2D>(mTopTexGuid);
        break;
    case CubemapFace::NegativeZ:
        texture = mWorld->getAssetByGuid<Texture2D>(mBackTexGuid);
        break;
    case CubemapFace::PositiveZ:
        texture = mWorld->getAssetByGuid<Texture2D>(mFrontTexGuid);
        break;
    }

    if (texture != nullptr)
    {
        this->setRawCubemapData(face, texture->getRawTextureData());
    }
}

void Cubemap::copyTextureToDevice()
{
    if (mDeviceUpdateRequired)
    {
        mCube->load(mFormat, mWrapMode, mFilterMode, mWidth, mRawTextureData);
        mDeviceUpdateRequired = false;
    }
}

void Cubemap::updateTextureParameters()
{
    if (mUpdateRequired)
    {
        mCube->update(mWrapMode, mFilterMode);
        mUpdateRequired = false;
    }
}

void Cubemap::readPixels()
{
    mCube->readPixels(mRawTextureData);
}

void Cubemap::writePixels()
{
    mCube->writePixels(mRawTextureData);
}

CubemapHandle *Cubemap::getNativeGraphics() const
{
    return mCube;
}