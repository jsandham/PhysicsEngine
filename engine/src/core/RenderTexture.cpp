#include <fstream>

#include "../../include/core/SerializationYaml.h"
#include "../../include/core/AssetYaml.h"
#include "../../include/core/RenderTexture.h"
#include "../../include/core/Log.h"
#include "../../include/core/World.h"

#include "../../include/graphics/Renderer.h"

#include "stb_image_write.h"

using namespace PhysicsEngine;

static int calcNumChannels(TextureFormat format)
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

RenderTexture::RenderTexture(World *world, const Id &id) : mWorld(world), mGuid(Guid::INVALID), mId(id), mHide(HideFlag::None)
{
    mName = "Unnamed Asset";

    mWidth = 1920;
    mHeight = 1080;
    mFormat = TextureFormat::RGBA;
    mWrapMode = TextureWrapMode::Repeat;
    mFilterMode = TextureFilterMode::Trilinear;

    mTargets.mMainFBO = Framebuffer::create(mWidth, mHeight);

    mNumChannels = calcNumChannels(mFormat);
    mAnisoLevel = 1;
    mDeviceUpdateRequired = false;
}

RenderTexture::RenderTexture(World *world, const Guid &guid, const Id &id) : mWorld(world), mGuid(guid), mId(id), mHide(HideFlag::None)
{
    mName = "Unnamed Asset";

    mWidth = 1920;
    mHeight = 1080;
    mFormat = TextureFormat::RGBA;
    mWrapMode = TextureWrapMode::Repeat;
    mFilterMode = TextureFilterMode::Trilinear;

    mTargets.mMainFBO = Framebuffer::create(mWidth, mHeight);

    mNumChannels = calcNumChannels(mFormat);
    mAnisoLevel = 1;
    mDeviceUpdateRequired = false;
}

RenderTexture::RenderTexture(World *world, const Id &id, int width, int height) : mWorld(world), mGuid(Guid::INVALID), mId(id), mHide(HideFlag::None)
{
    mName = "Unnamed Asset";

    mWidth = width;
    mHeight = height;
    mFormat = TextureFormat::RGBA;
    mWrapMode = TextureWrapMode::Repeat;
    mFilterMode = TextureFilterMode::Trilinear;

    mTargets.mMainFBO = Framebuffer::create(mWidth, mHeight);

    mNumChannels = calcNumChannels(mFormat);
    mAnisoLevel = 1;
    mDeviceUpdateRequired = false;

    mRawTextureData.resize(width * height * mNumChannels);
}

RenderTexture::RenderTexture(World *world, const Id &id, int width, int height, TextureFormat format) : mWorld(world), mId(id), mHide(HideFlag::None)
{
    mName = "Unnamed Asset";

    mWidth = width;
    mHeight = height;
    mFormat = format;
    mWrapMode = TextureWrapMode::Repeat;
    mFilterMode = TextureFilterMode::Trilinear;

    mTargets.mMainFBO = Framebuffer::create(mWidth, mHeight);

    mNumChannels = calcNumChannels(mFormat);
    mAnisoLevel = 1;
    mDeviceUpdateRequired = false;

    mRawTextureData.resize(width * height * mNumChannels);
}

RenderTexture::~RenderTexture()
{
    delete mTargets.mMainFBO;
}

void RenderTexture::serialize(YAML::Node &out) const
{
    out["type"] = getType();
    out["hide"] = mHide;
    out["id"] = mGuid;

    out["name"] = mName;

    out["dimension"] = mDimension;
    out["format"] = mFormat;
    out["wrapMode"] = mWrapMode;
    out["filterMode"] = mFilterMode;
    out["numChannels"] = mNumChannels;
    out["anisoLevel"] = mAnisoLevel;

    out["width"] = mWidth;
    out["height"] = mHeight;
}

void RenderTexture::deserialize(const YAML::Node &in)
{
    mHide = YAML::getValue<HideFlag>(in, "hide");
    mGuid = YAML::getValue<Guid>(in, "id");

    mName = YAML::getValue<std::string>(in, "name");

    mDimension = YAML::getValue<TextureDimension>(in, "dimension");
    mFormat = YAML::getValue<TextureFormat>(in, "format");
    mWrapMode = YAML::getValue<TextureWrapMode>(in, "wrapMode");
    mFilterMode = YAML::getValue<TextureFilterMode>(in, "filterMode");
    mNumChannels = YAML::getValue<int>(in, "numChannels");
    mAnisoLevel = YAML::getValue<int>(in, "anisoLevel");

    mDeviceUpdateRequired = false;
    mUpdateRequired = false;

    mWidth = YAML::getValue<int>(in, "width");
    mHeight = YAML::getValue<int>(in, "height");
}

bool RenderTexture::writeToYAML(const std::string &filepath) const
{
    std::ofstream out;
    out.open(filepath);

    if (!out.is_open())
    {
        return false;
    }

    if (mHide == HideFlag::None)
    {
        YAML::Node n;
        serialize(n);

        YAML::Node assetNode;
        assetNode[getObjectName()] = n;

        out << assetNode;
        out << "\n";
    }
    out.close();

    return true;
}

void RenderTexture::loadFromYAML(const std::string &filepath)
{
    YAML::Node in = YAML::LoadFile(filepath);

    if (!in.IsMap())
    {
        return;
    }

    for (YAML::const_iterator it = in.begin(); it != in.end(); ++it)
    {
        if (it->first.IsScalar() && it->second.IsMap())
        {
            deserialize(it->second);
        }
    }
}

int RenderTexture::getType() const
{
    return PhysicsEngine::RENDER_TEXTURE_TYPE;
}

std::string RenderTexture::getObjectName() const
{
    return PhysicsEngine::RENDER_TEXTURE_NAME;
}

Guid RenderTexture::getGuid() const
{
    return mGuid;
}

Id RenderTexture::getId() const
{
    return mId;
}

void RenderTexture::writeToPNG(const std::string &filepath) const
{
    int success = stbi_write_png(filepath.c_str(), mWidth, mHeight, mNumChannels, mRawTextureData.data(), mWidth);
    if (!success)
    {
        std::string message = "Error: stbi_write_png failed to write texture " + filepath + "\n";
        Log::error(message.c_str());
        return;
    }
}

void RenderTexture::writeToJPG(const std::string &filepath) const
{
    int success = stbi_write_jpg(filepath.c_str(), mWidth, mHeight, mNumChannels, mRawTextureData.data(), 100);
    if (!success)
    {
        std::string message = "Error: stbi_write_jpg failed to write texture " + filepath + "\n";
        Log::error(message.c_str());
        return;
    }
}

void RenderTexture::writeToBMP(const std::string &filepath) const
{
    int success = stbi_write_bmp(filepath.c_str(), mWidth, mHeight, mNumChannels, mRawTextureData.data());
    if (!success)
    {
        std::string message = "Error: stbi_write_bmp failed to write texture " + filepath + "\n";
        Log::error(message.c_str());
        return;
    }
}

int RenderTexture::getWidth() const
{
    return mWidth;
}

int RenderTexture::getHeight() const
{
    return mHeight;
}

void RenderTexture::copyTextureToDevice()
{
}

void RenderTexture::updateTextureParameters()
{
}

void RenderTexture::readPixels()
{
}

void RenderTexture::writePixels()
{
}

Framebuffer *RenderTexture::getNativeGraphicsMainFBO() const
{
    return mTargets.mMainFBO;
}

RenderTextureHandle *RenderTexture::getNativeGraphicsColorTex() const
{
    return mTargets.mMainFBO->getColorTex();
}

RenderTextureHandle *RenderTexture::getNativeGraphicsDepthTex() const
{
    return mTargets.mMainFBO->getDepthTex();
}