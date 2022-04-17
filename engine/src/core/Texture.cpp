#include "../../include/core/Texture.h"
#include "../../include/core/GLM.h"
#include "../../include/core/Log.h"

using namespace PhysicsEngine;

Texture::Texture(World *world) : Asset(world)
{
}

Texture::Texture(World *world, Id id) : Asset(world, id)
{
}

Texture::~Texture()
{
}

void Texture::serialize(YAML::Node &out) const
{
    Asset::serialize(out);

    out["dimension"] = mDimension;
    out["format"] = mFormat;
    out["wrapMode"] = mWrapMode;
    out["filterMode"] = mFilterMode;
    out["numChannels"] = mNumChannels;
    out["anisoLevel"] = mAnisoLevel;
}

void Texture::deserialize(const YAML::Node &in)
{
    Asset::deserialize(in);

    mDimension = YAML::getValue<TextureDimension>(in, "dimension");
    mFormat = YAML::getValue<TextureFormat>(in, "format");
    mWrapMode = YAML::getValue<TextureWrapMode>(in, "wrapMode");
    mFilterMode = YAML::getValue<TextureFilterMode>(in, "filterMode");
    mNumChannels = YAML::getValue<int>(in, "numChannels");
    mAnisoLevel = YAML::getValue<int>(in, "anisoLevel");

    mCreated = false;
    mUpdateRequired = false;
}

bool Texture::isCreated() const
{
    return mCreated;
}

bool Texture::updateRequired() const
{
    return mUpdateRequired;
}

int Texture::getNumChannels() const
{
    return mNumChannels;
}

int Texture::getAnisoLevel() const
{
    return mAnisoLevel;
}

TextureDimension Texture::getDimension() const
{
    return mDimension;
}

TextureFormat Texture::getFormat() const
{
    return mFormat;
}

TextureWrapMode Texture::getWrapMode() const
{
    return mWrapMode;
}

TextureFilterMode Texture::getFilterMode() const
{
    return mFilterMode;
}

unsigned int Texture::getNativeGraphics() const
{
    return mTex;
}

void Texture::setAnisoLevel(int anisoLevel)
{
    mAnisoLevel = anisoLevel;
    mUpdateRequired = true;
}

void Texture::setWrapMode(TextureWrapMode wrapMode)
{
    mWrapMode = wrapMode;
    mUpdateRequired = true;
}

void Texture::setFilterMode(TextureFilterMode filterMode)
{
    mFilterMode = filterMode;
    mUpdateRequired = true;
}

int Texture::calcNumChannels(TextureFormat format) const
{
    int nChannels = 0;

    switch (format)
    {
    case TextureFormat::Depth:
        nChannels = 1;
        break;
    case TextureFormat::RG:
        nChannels = 2;
        break;
    case TextureFormat::RGB:
        nChannels = 3;
        break;
    case TextureFormat::RGBA:
        nChannels = 4;
        break;
    default:
        Log::error("Error: Texture: Invalid texture format\n");
    }

    return nChannels;
}