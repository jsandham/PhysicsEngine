#include "../../include/core/RenderTexture.h"
#include "../../include/core/Log.h"
#include "../../include/graphics/Graphics.h"

#include "stb_image_write.h"

using namespace PhysicsEngine;

RenderTexture::RenderTexture(World* world) : Texture(world)
{
    mWidth = 1920;
    mHeight = 1080;
    mFormat = TextureFormat::RGBA;
    mWrapMode = TextureWrapMode::Repeat;
    mFilterMode = TextureFilterMode::Trilinear;

    mNumChannels = calcNumChannels(mFormat);
    mAnisoLevel = 1;
    mCreated = false;
}

RenderTexture::RenderTexture(World* world, Guid id) : Texture(world, id)
{
    mWidth = 1920;
    mHeight = 1080;
    mFormat = TextureFormat::RGBA;
    mWrapMode = TextureWrapMode::Repeat;
    mFilterMode = TextureFilterMode::Trilinear;

    mNumChannels = calcNumChannels(mFormat);
    mAnisoLevel = 1;
    mCreated = false;
}

RenderTexture::RenderTexture(World* world, int width, int height) : Texture(world)
{
    mWidth = width;
    mHeight = height;
    mFormat = TextureFormat::RGBA;
    mWrapMode = TextureWrapMode::Repeat;
    mFilterMode = TextureFilterMode::Trilinear;

    mNumChannels = calcNumChannels(mFormat);
    mAnisoLevel = 1;
    mCreated = false;

    mRawTextureData.resize(width * height * mNumChannels);
}

RenderTexture::RenderTexture(World* world, int width, int height, TextureFormat format) : Texture(world)
{
    mWidth = width;
    mHeight = height;
    mFormat = format;
    mWrapMode = TextureWrapMode::Repeat;
    mFilterMode = TextureFilterMode::Trilinear;

    mNumChannels = calcNumChannels(mFormat);
    mAnisoLevel = 1;
    mCreated = false;

    mRawTextureData.resize(width * height * mNumChannels);
}

RenderTexture::~RenderTexture()
{

}

void RenderTexture::serialize(YAML::Node& out) const
{
    Texture::serialize(out);

    out["width"] = mWidth;
    out["height"] = mHeight;
}

void RenderTexture::deserialize(const YAML::Node& in)
{
    Texture::deserialize(in);

    mWidth = YAML::getValue<int>(in, "width");
    mHeight = YAML::getValue<int>(in, "height");
}

int RenderTexture::getType() const
{
    return PhysicsEngine::RENDER_TEXTURE_TYPE;
}

std::string RenderTexture::getObjectName() const
{
    return PhysicsEngine::RENDER_TEXTURE_NAME;
}

void RenderTexture::writeToPNG(const std::string& filepath) const
{
    int success = stbi_write_png(filepath.c_str(), mWidth, mHeight, mNumChannels, mRawTextureData.data(), mWidth);
    if (!success)
    {
        std::string message = "Error: stbi_write_png failed to write texture " + filepath + "\n";
        Log::error(message.c_str());
        return;
    }
}

void RenderTexture::writeToJPG(const std::string& filepath) const
{
    int success = stbi_write_jpg(filepath.c_str(), mWidth, mHeight, mNumChannels, mRawTextureData.data(), 100);
    if (!success)
    {
        std::string message = "Error: stbi_write_jpg failed to write texture " + filepath + "\n";
        Log::error(message.c_str());
        return;
    }
}

void RenderTexture::writeToBMP(const std::string& filepath) const
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

void RenderTexture::create()
{
    if (mCreated)
    {
        return;
    }

    Graphics::createRenderTextureTargets(&mTargets, mFormat, mWrapMode, mFilterMode, mWidth, mHeight);

    mCreated = true;
}

void RenderTexture::destroy()
{
    if (!mCreated)
    {
        return;
    }

    Graphics::destroyRenderTextureTargets(&mTargets);

    mCreated = false;
}

void RenderTexture::update()
{

}

void RenderTexture::readPixels()
{

}

void RenderTexture::writePixels()
{

}

unsigned int RenderTexture::getNativeGraphicsMainFBO() const
{
    return mTargets.mMainFBO;
}

unsigned int RenderTexture::getNativeGraphicsColorTex() const
{
    return mTargets.mColorTex;
}

unsigned int RenderTexture::getNativeGraphicsDepthTex() const
{
    return mTargets.mDepthTex;
}