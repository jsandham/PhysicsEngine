#include "../../include/core/Texture2D.h"
#include "../../include/core/Log.h"
#include "../../include/core/Serialization.h"
#include "../../include/graphics/Graphics.h"
#include "stb_image.h"
#include "stb_image_write.h"

using namespace PhysicsEngine;

Texture2D::Texture2D(World *world) : Texture(world)
{
    mDimension = TextureDimension::Tex2D;

    mSource = "";
    mWidth = 0;
    mHeight = 0;
    mFormat = TextureFormat::RGB;
    mWrapMode = TextureWrapMode::Repeat;
    mFilterMode = TextureFilterMode::Trilinear;

    mNumChannels = calcNumChannels(mFormat);
    mAnisoLevel = 1;
    mCreated = false;
    mUpdateRequired = false;
}

Texture2D::Texture2D(World *world, Guid id) : Texture(world, id)
{
    mDimension = TextureDimension::Tex2D;

    mSource = "";
    mWidth = 0;
    mHeight = 0;
    mFormat = TextureFormat::RGB;
    mWrapMode = TextureWrapMode::Repeat;
    mFilterMode = TextureFilterMode::Trilinear;

    mNumChannels = calcNumChannels(mFormat);
    mAnisoLevel = 1;
    mCreated = false;
    mUpdateRequired = false;
}

Texture2D::Texture2D(World *world, int width, int height) : Texture(world)
{
    mDimension = TextureDimension::Tex2D;

    mSource = "";
    mWidth = width;
    mHeight = height;
    mFormat = TextureFormat::RGB;
    mWrapMode = TextureWrapMode::Repeat;
    mFilterMode = TextureFilterMode::Trilinear;

    mNumChannels = calcNumChannels(mFormat);
    mAnisoLevel = 1;
    mCreated = false;
    mUpdateRequired = false;

    mRawTextureData.resize(width * height * mNumChannels);
}

Texture2D::Texture2D(World *world, int width, int height, TextureFormat format) : Texture(world)
{
    mDimension = TextureDimension::Tex2D;

    mSource = "";
    mWidth = width;
    mHeight = height;
    mFormat = format;

    mWrapMode = TextureWrapMode::Repeat;
    mFilterMode = TextureFilterMode::Trilinear;

    mNumChannels = calcNumChannels(format);
    mAnisoLevel = 1;
    mCreated = false;
    mUpdateRequired = false;

    mRawTextureData.resize(width * height * mNumChannels);
}

Texture2D::~Texture2D()
{
}

void Texture2D::serialize(YAML::Node &out) const
{
    Texture::serialize(out);

    out["width"] = mWidth;
    out["height"] = mHeight;
    out["source"] = mSource;
}

void Texture2D::deserialize(const YAML::Node &in)
{
    Texture::deserialize(in);

    mWidth = YAML::getValue<int>(in, "width");
    mHeight = YAML::getValue<int>(in, "height");

    mSource = YAML::getValue<std::string>(in, "source");
    load(mSource);
}

int Texture2D::getType() const
{
    return PhysicsEngine::TEXTURE2D_TYPE;
}

std::string Texture2D::getObjectName() const
{
    return PhysicsEngine::TEXTURE2D_NAME;
}

void Texture2D::load(const std::string &filepath)
{
    if (filepath.empty())
    {
        return;
    }

    stbi_set_flip_vertically_on_load(true);

    int width, height, numChannels;
    unsigned char *raw = stbi_load(filepath.c_str(), &width, &height, &numChannels, 0);

    if (raw != NULL)
    {
        TextureFormat format;
        switch (numChannels)
        {
        case 1:
            format = TextureFormat::Depth;
            break;
        case 2:
            format = TextureFormat::RG;
            break;
        case 3:
            format = TextureFormat::RGB;
            break;
        case 4:
            format = TextureFormat::RGBA;
            break;
        default:
            std::string message = "Error: Unsupported number of channels (" + std::to_string(numChannels) +
                                  ") found when loading texture " + filepath + "\n";
            Log::error(message.c_str());
            return;
        }

        std::vector<unsigned char> data;
        data.resize(width * height * numChannels);

        for (size_t j = 0; j < data.size(); j++)
        {
            data[j] = raw[j];
        }

        stbi_image_free(raw);

        setRawTextureData(data, width, height, format);
    }
    else
    {
        std::string message = "Error: stbi_load failed to load texture " + filepath +
                              " with reported reason: " + stbi_failure_reason() + "\n";
        Log::error(message.c_str());
        return;
    }

    mSource = filepath;
}

void Texture2D::writeToPNG(const std::string &filepath) const
{
    int success = stbi_write_png(filepath.c_str(), mWidth, mHeight, mNumChannels, mRawTextureData.data(), mWidth);
    if (!success)
    {
        std::string message = "Error: stbi_write_png failed to write texture " + filepath + "\n";
        Log::error(message.c_str());
        return;
    }
}

void Texture2D::writeToJPG(const std::string &filepath) const
{
    int success = stbi_write_jpg(filepath.c_str(), mWidth, mHeight, mNumChannels, mRawTextureData.data(), 100);
    if (!success)
    {
        std::string message = "Error: stbi_write_jpg failed to write texture " + filepath + "\n";
        Log::error(message.c_str());
        return;
    }
}

void Texture2D::writeToBMP(const std::string &filepath) const
{
    int success = stbi_write_bmp(filepath.c_str(), mWidth, mHeight, mNumChannels, mRawTextureData.data());
    if (!success)
    {
        std::string message = "Error: stbi_write_bmp failed to write texture " + filepath + "\n";
        Log::error(message.c_str());
        return;
    }
}

int Texture2D::getWidth() const
{
    return mWidth;
}

int Texture2D::getHeight() const
{
    return mHeight;
}

void Texture2D::redefine(int width, int height, TextureFormat format)
{
    mWidth = width;
    mHeight = height;
    mFormat = format;

    mNumChannels = calcNumChannels(format);

    mRawTextureData.resize(width * height * mNumChannels);
}

std::vector<unsigned char> Texture2D::getRawTextureData() const
{
    return mRawTextureData;
}

std::vector<Color32> Texture2D::getPixels() const
{
    std::vector<Color32> colors;

    colors.resize(mWidth * mHeight);

    for (unsigned int i = 0; i < colors.size(); i++)
    {
        colors[i].mR = mRawTextureData[mNumChannels * i];
        colors[i].mG = mRawTextureData[mNumChannels * i + 1];
        colors[i].mB = mRawTextureData[mNumChannels * i + 2];
    }

    return colors;
}

Color32 Texture2D::getPixel(int x, int y) const
{
    // clamp x and y
    x = std::max(0, std::min(mWidth, x));
    y = std::max(0, std::min(mHeight, y));

    int index = mNumChannels * (x + mWidth * y);

    Color32 color;

    int size = static_cast<int>(mRawTextureData.size());
    if (index + mNumChannels >= size)
    {
        Log::error("Texture2D: pixel index out of range\n");
        return color;
    }

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

void Texture2D::setRawTextureData(const std::vector<unsigned char> &data, int width, int height, TextureFormat format)
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
    mFormat = format;

    mRawTextureData = data;
}

void Texture2D::setPixels(const std::vector<Color32> &colors)
{
    int size = static_cast<int>(mRawTextureData.size());
    if (size != mWidth * mHeight)
    {
        Log::error("Texture2D: error when trying to set pixels. Number of colors must match dimensions of texture\n");
        return;
    }

    for (size_t i = 0; i < colors.size(); i++)
    {
        if (mNumChannels == 1)
        {
            mRawTextureData[i] = colors[i].mR;
        }
        else if (mNumChannels == 3)
        {
            mRawTextureData[i] = colors[i].mR;
            mRawTextureData[i + 1] = colors[i].mG;
            mRawTextureData[i + 2] = colors[i].mB;
        }
        else if (mNumChannels == 4)
        {
            mRawTextureData[i] = colors[i].mR;
            mRawTextureData[i + 1] = colors[i].mG;
            mRawTextureData[i + 2] = colors[i].mB;
            mRawTextureData[i + 3] = colors[i].mA;
        }
    }
}

void Texture2D::setPixel(int x, int y, const Color32 &color)
{
    // clamp x and y
    x = std::max(0, std::min(mWidth, x));
    y = std::max(0, std::min(mHeight, y));

    int index = mNumChannels * (x + mWidth * y);

    int size = static_cast<int>(mRawTextureData.size());
    if (index + mNumChannels >= size)
    {
        Log::error("Texture2D: pixel index out of range\n");
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

void Texture2D::create()
{
    if (mCreated)
    {
        return;
    }

    Graphics::createTexture2D(mFormat, mWrapMode, mFilterMode, mWidth, mHeight, mRawTextureData, &mTex);

    mCreated = true;
}

void Texture2D::destroy()
{
    if (!mCreated)
    {
        return;
    }

    Graphics::destroyTexture2D(&mTex);

    mCreated = false;
}

void Texture2D::update()
{
    if (!mUpdateRequired)
    {
        return;
    }

    Graphics::updateTexture2D(mWrapMode, mFilterMode, mAnisoLevel, mTex);

    mUpdateRequired = false;
}

void Texture2D::readPixels()
{
    Graphics::readPixelsTexture2D(mFormat, mWidth, mHeight, mNumChannels, mRawTextureData, mTex);
}

void Texture2D::writePixels()
{
    Graphics::writePixelsTexture2D(mFormat, mWidth, mHeight, mRawTextureData, mTex);
}