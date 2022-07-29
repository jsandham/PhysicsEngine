#include "../../include/core/Texture3D.h"
#include "../../include/core/Log.h"
#include "../../include/graphics/Graphics.h"

using namespace PhysicsEngine;

Texture3D::Texture3D(World *world, const Id &id) : Texture(world, id)
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

Texture3D::Texture3D(World *world, const Guid &guid, const Id &id) : Texture(world, guid, id)
{
    mDimension = TextureDimension::Tex3D;

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

Texture3D::Texture3D(World *world, const Id &id, int width, int height, int depth, int numChannels) : Texture(world, id)
{
    mDimension = TextureDimension::Tex3D;

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

void Texture3D::serialize(YAML::Node &out) const
{
    Texture::serialize(out);

    out["width"] = mWidth;
    out["height"] = mHeight;
    out["depth"] = mDepth;
}

void Texture3D::deserialize(const YAML::Node &in)
{
    Texture::deserialize(in);

    mWidth = YAML::getValue<int>(in, "width");
    mHeight = YAML::getValue<int>(in, "height");
    mDepth = YAML::getValue<int>(in, "depth");
}

int Texture3D::getType() const
{
    return PhysicsEngine::TEXTURE3D_TYPE;
}

std::string Texture3D::getObjectName() const
{
    return PhysicsEngine::TEXTURE3D_NAME;
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
