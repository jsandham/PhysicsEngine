#include "../../include/core/Texture.h"
#include "../../include/core/Log.h"
#include "../../include/core/Serialization.h"

using namespace PhysicsEngine;

Texture::Texture() : Asset()
{
}

Texture::Texture(Guid id) : Asset(id)
{
}

Texture::~Texture()
{
}

void Texture::serialize(std::ostream &out) const
{
    Asset::serialize(out);

    PhysicsEngine::write<TextureDimension>(out, mDimension);
    PhysicsEngine::write<TextureFormat>(out, mFormat);
    PhysicsEngine::write<TextureWrapMode>(out, mWrapMode);
    PhysicsEngine::write<TextureFilterMode>(out, mFilterMode);
    PhysicsEngine::write<int>(out, mNumChannels);
    PhysicsEngine::write<int>(out, mAnisoLevel);
    PhysicsEngine::write<size_t>(out, mRawTextureData.size());
    PhysicsEngine::write<const unsigned char>(out, mRawTextureData.data(), mRawTextureData.size());
}

void Texture::deserialize(std::istream &in)
{
    Asset::deserialize(in);

    size_t dataSize;
    PhysicsEngine::read<TextureDimension>(in, mDimension);
    PhysicsEngine::read<TextureFormat>(in, mFormat);
    PhysicsEngine::read<TextureWrapMode>(in, mWrapMode);
    PhysicsEngine::read<TextureFilterMode>(in, mFilterMode);
    PhysicsEngine::read<int>(in, mNumChannels);
    PhysicsEngine::read<int>(in, mAnisoLevel);
    PhysicsEngine::read<size_t>(in, dataSize);
    PhysicsEngine::read<unsigned char>(in, mRawTextureData.data(), dataSize);

    mCreated = false;
    mUpdateRequired = false;
}

void Texture::serialize(YAML::Node& out) const
{
    Asset::serialize(out);

    out["dimension"] = mDimension;
    out["format"] = mFormat;
    out["wrapMode"] = mWrapMode;
    out["filterMode"] = mFilterMode;
    out["numChannels"] = mNumChannels;
    out["anisoLevel"] = mAnisoLevel;
}

void Texture::deserialize(const YAML::Node& in)
{
    Asset::deserialize(in);

    mDimension = in["dimension"].as<TextureDimension>();
    mFormat = in["format"].as<TextureFormat>();
    mWrapMode = in["wrapMode"].as<TextureWrapMode>();
    mFilterMode = in["filterMode"].as<TextureFilterMode>();
    mNumChannels = in["numChannels"].as<int>();
    mAnisoLevel = in["anisoLevel"].as<int>();

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

GLuint Texture::getNativeGraphics() const
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