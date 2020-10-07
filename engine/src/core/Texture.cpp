#include "../../include/core/Texture.h"
#include "../../include/core/Log.h"

using namespace PhysicsEngine;

Texture::Texture()
{
    mAssetId = Guid::INVALID;
}

bool Texture::isCreated() const
{
    return mCreated;
}

int Texture::getNumChannels() const
{
    return mNumChannels;
}

TextureDimension Texture::getDimension() const
{
    return mDimension;
}

TextureFormat Texture::getFormat() const
{
    return mFormat;
}

GLuint Texture::getNativeGraphics() const
{
    return mTex;
}

int Texture::calcNumChannels(TextureFormat format) const
{
    int nChannels = 0;

    switch (format)
    {
    case Depth:
        nChannels = 1;
        break;
    case RG:
        nChannels = 2;
        break;
    case RGB:
        nChannels = 3;
        break;
    case RGBA:
        nChannels = 4;
        break;
    default:
        Log::error("Error: Texture: Invalid texture format\n");
    }

    return nChannels;
}